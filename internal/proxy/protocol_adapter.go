package proxy

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// ProtocolAdapter converts request/response payloads and paths between different channel protocols
// so that aggregate groups can mix sub-groups using heterogeneous channel types.
//
// Current implementation focuses on chat-style endpoints for OpenAI-like and Anthropic-compatible
// protocols. Requests are normalized into a lightweight internal representation and then rendered
// back to the target protocol. Unsupported protocols or payload shapes return an error so callers
// can surface a clear message to clients.
type ProtocolAdapter struct{}

// NewProtocolAdapter constructs a new adapter instance.
func NewProtocolAdapter() *ProtocolAdapter {
	return &ProtocolAdapter{}
}

// AdaptRequest converts the incoming payload and path (written for sourceType) so it can be sent to
// a targetType upstream. Returns the rewritten path (empty string means unchanged) and the new body
// bytes. Only chat-style requests are supported; streaming conversions are intentionally rejected at
// the call site.
func (a *ProtocolAdapter) AdaptRequest(sourceType, targetType, path string, body []byte) (string, []byte, error) {
	if sourceType == targetType {
		return "", body, nil
	}

	chatReq, err := normalizeChatRequest(sourceType, body)
	if err != nil {
		return "", nil, fmt.Errorf("failed to normalize %s request: %w", sourceType, err)
	}

	newBody, err := renderChatRequest(targetType, chatReq)
	if err != nil {
		return "", nil, fmt.Errorf("failed to render %s request: %w", targetType, err)
	}

	targetPath := defaultChatPath(targetType)
	if targetPath == "" {
		return "", nil, fmt.Errorf("unsupported target channel type: %s", targetType)
	}

	return targetPath, newBody, nil
}

// AdaptResponse converts the upstream response (produced by targetType) back into the original
// sourceType protocol so clients see the format they expect.
func (a *ProtocolAdapter) AdaptResponse(sourceType, targetType string, body []byte) ([]byte, string, error) {
	if sourceType == targetType {
		return body, "", nil
	}

	chatResp, err := normalizeChatResponse(targetType, body)
	if err != nil {
		return nil, "", fmt.Errorf("failed to normalize %s response: %w", targetType, err)
	}

	adaptedBody, err := renderChatResponse(sourceType, chatResp)
	if err != nil {
		return nil, "", fmt.Errorf("failed to render %s response: %w", sourceType, err)
	}

	return adaptedBody, "application/json", nil
}

// AdaptStream relays and adapts streaming responses (SSE) produced by targetType so
// they appear in the format expected by sourceType.
func (a *ProtocolAdapter) AdaptStream(sourceType, targetType string, upstream io.Reader, downstream io.Writer, flusher http.Flusher) error {
	if sourceType == targetType {
		_, err := io.Copy(downstream, upstream)
		return err
	}

	// Only OpenAI <-> Anthropic chat streaming is supported.
	src := strings.ToLower(sourceType)
	tgt := strings.ToLower(targetType)

	switch {
	case src == "openai" && tgt == "anthropic":
		return adaptAnthropicStreamToOpenAI(upstream, downstream, flusher)
	case src == "anthropic" && tgt == "openai":
		return adaptOpenAIStreamToAnthropic(upstream, downstream, flusher)
	default:
		return fmt.Errorf("cross-channel streaming between %s and %s is not supported", sourceType, targetType)
	}
}

type chatMessage struct {
	Role    string
	Content string
}

type chatRequest struct {
	Model       string
	Messages    []chatMessage
	Stream      bool
	Temperature *float64
	MaxTokens   *int
}

type chatResponse struct {
	ID           string
	Model        string
	Message      chatMessage
	FinishReason string
	Created      int64
	Usage        map[string]any
}

// normalizeChatRequest parses a client payload into a neutral chatRequest.
func normalizeChatRequest(channelType string, body []byte) (*chatRequest, error) {
	switch strings.ToLower(channelType) {
	case "openai":
		return parseOpenAIChatRequest(body)
	case "anthropic":
		return parseAnthropicChatRequest(body)
	default:
		return nil, fmt.Errorf("channel %s is not supported for cross-protocol adaptation", channelType)
	}
}

func renderChatRequest(channelType string, req *chatRequest) ([]byte, error) {
	switch strings.ToLower(channelType) {
	case "openai":
		return renderOpenAIChatRequest(req)
	case "anthropic":
		return renderAnthropicChatRequest(req)
	default:
		return nil, fmt.Errorf("channel %s is not supported for cross-protocol adaptation", channelType)
	}
}

func normalizeChatResponse(channelType string, body []byte) (*chatResponse, error) {
	switch strings.ToLower(channelType) {
	case "openai":
		return parseOpenAIChatResponse(body)
	case "anthropic":
		return parseAnthropicChatResponse(body)
	default:
		return nil, fmt.Errorf("channel %s is not supported for cross-protocol adaptation", channelType)
	}
}

func renderChatResponse(channelType string, resp *chatResponse) ([]byte, error) {
	switch strings.ToLower(channelType) {
	case "openai":
		return renderOpenAIChatResponse(resp)
	case "anthropic":
		return renderAnthropicChatResponse(resp)
	default:
		return nil, fmt.Errorf("channel %s is not supported for cross-protocol adaptation", channelType)
	}
}

func defaultChatPath(channelType string) string {
	switch strings.ToLower(channelType) {
	case "openai":
		return "/v1/chat/completions"
	case "anthropic":
		return "/v1/messages"
	default:
		return ""
	}
}

// --- OpenAI chat conversions ---

type openAIChatMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

type openAIChatRequest struct {
	Model       string              `json:"model"`
	Messages    []openAIChatMessage `json:"messages"`
	Stream      bool                `json:"stream,omitempty"`
	Temperature *float64            `json:"temperature,omitempty"`
	MaxTokens   *int                `json:"max_tokens,omitempty"`
}

func parseOpenAIChatRequest(body []byte) (*chatRequest, error) {
	var req openAIChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, err
	}

	msgs := make([]chatMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		text := extractOpenAIContent(m.Content)
		msgs = append(msgs, chatMessage{Role: m.Role, Content: text})
	}

	return &chatRequest{
		Model:       req.Model,
		Messages:    msgs,
		Stream:      req.Stream,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
	}, nil
}

func renderOpenAIChatRequest(req *chatRequest) ([]byte, error) {
	msgs := make([]openAIChatMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		msgs = append(msgs, openAIChatMessage{Role: m.Role, Content: m.Content})
	}

	out := openAIChatRequest{
		Model:       req.Model,
		Messages:    msgs,
		Stream:      req.Stream,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
	}
	return json.Marshal(out)
}

func parseOpenAIChatResponse(body []byte) (*chatResponse, error) {
	type choice struct {
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	}
	var resp struct {
		ID      string         `json:"id"`
		Model   string         `json:"model"`
		Created int64          `json:"created"`
		Choices []choice       `json:"choices"`
		Usage   map[string]any `json:"usage"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, err
	}

	if len(resp.Choices) == 0 {
		return nil, errors.New("no choices in OpenAI response")
	}

	first := resp.Choices[0]
	return &chatResponse{
		ID:           resp.ID,
		Model:        resp.Model,
		Created:      resp.Created,
		Message:      chatMessage{Role: first.Message.Role, Content: first.Message.Content},
		FinishReason: first.FinishReason,
		Usage:        resp.Usage,
	}, nil
}

func renderOpenAIChatResponse(resp *chatResponse) ([]byte, error) {
	out := map[string]any{
		"id":      resp.ID,
		"object":  "chat.completion",
		"model":   resp.Model,
		"created": resp.Created,
		"choices": []map[string]any{
			{
				"index":         0,
				"finish_reason": resp.FinishReason,
				"message": map[string]any{
					"role":    resp.Message.Role,
					"content": resp.Message.Content,
				},
			},
		},
	}
	if resp.Usage != nil {
		out["usage"] = resp.Usage
	}
	return json.Marshal(out)
}

func extractOpenAIContent(content interface{}) string {
	switch v := content.(type) {
	case string:
		return v
	case []any:
		var parts []string
		for _, item := range v {
			if m, ok := item.(map[string]any); ok {
				if text, ok := m["text"].(string); ok {
					parts = append(parts, text)
				}
			}
		}
		return strings.Join(parts, "")
	case map[string]any:
		if text, ok := v["text"].(string); ok {
			return text
		}
	}
	return ""
}

// --- Anthropic conversions ---

type anthropicContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type anthropicMessage struct {
	Role    string             `json:"role"`
	Content []anthropicContent `json:"content"`
}

type anthropicChatRequest struct {
	Model       string             `json:"model"`
	Messages    []anthropicMessage `json:"messages"`
	Stream      bool               `json:"stream,omitempty"`
	Temperature *float64           `json:"temperature,omitempty"`
	MaxTokens   *int               `json:"max_tokens,omitempty"`
}

func parseAnthropicChatRequest(body []byte) (*chatRequest, error) {
	var req anthropicChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, err
	}

	msgs := make([]chatMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		content := ""
		if len(m.Content) > 0 {
			content = m.Content[0].Text
		}
		msgs = append(msgs, chatMessage{Role: m.Role, Content: content})
	}

	return &chatRequest{
		Model:       req.Model,
		Messages:    msgs,
		Stream:      req.Stream,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
	}, nil
}

func renderAnthropicChatRequest(req *chatRequest) ([]byte, error) {
	msgs := make([]anthropicMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		msgs = append(msgs, anthropicMessage{
			Role: m.Role,
			Content: []anthropicContent{
				{Type: "text", Text: m.Content},
			},
		})
	}

	out := anthropicChatRequest{
		Model:       req.Model,
		Messages:    msgs,
		Stream:      req.Stream,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
	}
	return json.Marshal(out)
}

func parseAnthropicChatResponse(body []byte) (*chatResponse, error) {
	var resp struct {
		ID           string             `json:"id"`
		Model        string             `json:"model"`
		CreatedAt    int64              `json:"created_at"`
		Content      []anthropicContent `json:"content"`
		StopReason   string             `json:"stop_reason"`
		Usage        map[string]any     `json:"usage"`
		Type         string             `json:"type"`
		StopSequence string             `json:"stop_sequence"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, err
	}

	content := ""
	if len(resp.Content) > 0 {
		content = resp.Content[0].Text
	}

	finish := resp.StopReason
	if finish == "stop_sequence" {
		finish = resp.StopSequence
	}

	return &chatResponse{
		ID:           resp.ID,
		Model:        resp.Model,
		Created:      resp.CreatedAt,
		Message:      chatMessage{Role: "assistant", Content: content},
		FinishReason: finish,
		Usage:        resp.Usage,
	}, nil
}

func renderAnthropicChatResponse(resp *chatResponse) ([]byte, error) {
	out := map[string]any{
		"id":    resp.ID,
		"type":  "message",
		"model": resp.Model,
		"content": []map[string]any{
			{
				"type": "text",
				"text": resp.Message.Content,
			},
		},
		"stop_reason":   resp.FinishReason,
		"stop_sequence": resp.FinishReason,
	}
	if resp.Created != 0 {
		out["created_at"] = resp.Created
	}
	if resp.Usage != nil {
		out["usage"] = resp.Usage
	}
	return json.Marshal(out)
}

// --- Streaming helpers ---

type sseEvent struct {
	Event string
	Data  string
}

func newSSEDecoder(r io.Reader) *sseDecoder {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	return &sseDecoder{scanner: scanner}
}

type sseDecoder struct {
	scanner *bufio.Scanner
	next    *sseEvent
	err     error
}

func (d *sseDecoder) Next() bool {
	var event, data string
	var dataLines []string

	for d.scanner.Scan() {
		line := d.scanner.Text()
		if strings.HasPrefix(line, "event:") {
			event = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			continue
		}
		if strings.HasPrefix(line, "data:") {
			dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
			continue
		}
		if line == "" {
			if len(dataLines) == 0 && event == "" {
				continue
			}
			data = strings.Join(dataLines, "\n")
			d.next = &sseEvent{Event: event, Data: data}
			return true
		}
	}

	if err := d.scanner.Err(); err != nil {
		d.err = err
		return false
	}

	if len(dataLines) > 0 || event != "" {
		data = strings.Join(dataLines, "\n")
		d.next = &sseEvent{Event: event, Data: data}
		return true
	}

	return false
}

func (d *sseDecoder) Event() *sseEvent {
	return d.next
}

func (d *sseDecoder) Err() error {
	if d.err != nil {
		return d.err
	}
	return d.scanner.Err()
}

func writeSSE(w io.Writer, flusher http.Flusher, event string, payload any) error {
	var data string
	switch v := payload.(type) {
	case string:
		data = v
	default:
		b, err := json.Marshal(v)
		if err != nil {
			return err
		}
		data = string(b)
	}

	if event != "" {
		if _, err := fmt.Fprintf(w, "event: %s\n", event); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
		return err
	}
	if flusher != nil {
		flusher.Flush()
	}
	return nil
}

type anthropicStreamEvent struct {
	Type      string `json:"type"`
	ID        string `json:"id,omitempty"`
	Model     string `json:"model,omitempty"`
	CreatedAt int64  `json:"created_at,omitempty"`
	Delta     *struct {
		Text string `json:"text"`
	} `json:"delta,omitempty"`
	Content    []anthropicContent `json:"content,omitempty"`
	StopReason string             `json:"stop_reason,omitempty"`
	Usage      map[string]any     `json:"usage,omitempty"`
}

type openAIStreamDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type openAIStreamChoice struct {
	Delta        openAIStreamDelta `json:"delta"`
	FinishReason *string           `json:"finish_reason"`
}

type openAIStreamChunk struct {
	ID      string               `json:"id"`
	Object  string               `json:"object"`
	Created int64                `json:"created"`
	Model   string               `json:"model"`
	Choices []openAIStreamChoice `json:"choices"`
	Usage   map[string]any       `json:"usage,omitempty"`
}

func adaptAnthropicStreamToOpenAI(upstream io.Reader, downstream io.Writer, flusher http.Flusher) error {
	decoder := newSSEDecoder(upstream)
	var id, model string
	created := time.Now().Unix()
	var usage map[string]any
	sentRole := false

	for decoder.Next() {
		ev := decoder.Event()
		if ev == nil {
			continue
		}

		var payload anthropicStreamEvent
		if err := json.Unmarshal([]byte(ev.Data), &payload); err != nil {
			return err
		}

		switch payload.Type {
		case "message_start":
			if payload.ID != "" {
				id = payload.ID
			}
			if payload.Model != "" {
				model = payload.Model
			}
			if payload.CreatedAt != 0 {
				created = payload.CreatedAt
			}
		case "content_block_delta":
			text := ""
			if payload.Delta != nil {
				text = payload.Delta.Text
			}
			if text == "" && len(payload.Content) > 0 {
				text = payload.Content[0].Text
			}
			if text == "" {
				continue
			}

			role := ""
			if !sentRole {
				role = "assistant"
				sentRole = true
			}

			chunk := openAIStreamChunk{
				ID:      id,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []openAIStreamChoice{{
					Delta: openAIStreamDelta{
						Role:    role,
						Content: text,
					},
				}},
			}

			if err := writeSSE(downstream, flusher, "", chunk); err != nil {
				return err
			}
		case "message_delta":
			if payload.Usage != nil {
				usage = payload.Usage
			}
		case "message_stop":
			finish := payload.StopReason
			if finish == "" {
				finish = "stop"
			}
			final := openAIStreamChunk{
				ID:      id,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Usage:   usage,
				Choices: []openAIStreamChoice{{
					FinishReason: &finish,
				}},
			}
			if err := writeSSE(downstream, flusher, "", final); err != nil {
				return err
			}
			if err := writeSSE(downstream, flusher, "", "[DONE]"); err != nil {
				return err
			}
			return nil
		}
	}

	if err := decoder.Err(); err != nil {
		return err
	}
	return nil
}

func adaptOpenAIStreamToAnthropic(upstream io.Reader, downstream io.Writer, flusher http.Flusher) error {
	decoder := newSSEDecoder(upstream)
	var id, model string
	var created int64
	contentStarted := false
	for decoder.Next() {
		ev := decoder.Event()
		if ev == nil {
			continue
		}

		if ev.Data == "[DONE]" {
			stop := map[string]any{
				"type":        "message_stop",
				"stop_reason": "stop",
			}
			if err := writeSSE(downstream, flusher, "message_stop", stop); err != nil {
				return err
			}
			return nil
		}

		var payload openAIStreamChunk
		if err := json.Unmarshal([]byte(ev.Data), &payload); err != nil {
			return err
		}

		if id == "" {
			id = payload.ID
		}
		if model == "" {
			model = payload.Model
		}
		if created == 0 {
			created = payload.Created
			if created == 0 {
				created = time.Now().Unix()
			}
			start := map[string]any{
				"type":       "message_start",
				"id":         id,
				"model":      model,
				"created_at": created,
				"role":       "assistant",
			}
			if err := writeSSE(downstream, flusher, "message_start", start); err != nil {
				return err
			}
			if err := writeSSE(downstream, flusher, "content_block_start", map[string]any{"type": "content_block_start", "index": 0, "content_block": map[string]any{"type": "text"}}); err != nil {
				return err
			}
			contentStarted = true
		}

		if len(payload.Choices) == 0 {
			continue
		}
		choice := payload.Choices[0]

		if choice.Delta.Content != "" {
			delta := map[string]any{
				"type":  "content_block_delta",
				"index": 0,
				"delta": map[string]any{"type": "text_delta", "text": choice.Delta.Content},
			}
			if err := writeSSE(downstream, flusher, "content_block_delta", delta); err != nil {
				return err
			}
		}

		if choice.FinishReason != nil {
			if !contentStarted {
				// Ensure start events exist so downstream protocol remains valid
				if err := writeSSE(downstream, flusher, "content_block_start", map[string]any{"type": "content_block_start", "index": 0, "content_block": map[string]any{"type": "text"}}); err != nil {
					return err
				}
			}
			stop := map[string]any{
				"type":        "message_stop",
				"stop_reason": *choice.FinishReason,
			}
			if err := writeSSE(downstream, flusher, "message_stop", stop); err != nil {
				return err
			}
			return nil
		}
	}

	if err := decoder.Err(); err != nil {
		return err
	}
	return nil
}
