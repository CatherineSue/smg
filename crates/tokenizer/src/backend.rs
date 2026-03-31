//! Enum-based tokenizer dispatch — replaces `dyn Tokenizer` trait objects on the hot path.
//!
//! Using an enum instead of a vtable allows the compiler to inline decode calls
//! and enables variant-specific optimizations (e.g., native `DecodeStream` for HF).

use anyhow::Result;

use crate::{
    chat_template::{ChatTemplateContentFormat, ChatTemplateParams},
    huggingface::HuggingFaceTokenizer,
    mock::MockTokenizer,
    tiktoken::TiktokenTokenizer,
    traits::{self, Decoder, Encoding, SpecialTokens, TokenIdType},
};

/// Concrete tokenizer backend — one variant per supported tokenizer type.
///
/// Stored behind `Arc<TokenizerBackend>` and passed through the hot path
/// (Sequence, StopSequenceDecoder, DecodeStream). Enum dispatch replaces
/// vtable indirection, letting the compiler inline and optimize per-backend.
pub enum TokenizerBackend {
    HuggingFace(HuggingFaceTokenizer),
    Tiktoken(TiktokenTokenizer),
    Mock(MockTokenizer),
}

// ---------------------------------------------------------------------------
// Trait implementations — mechanical delegation to the inner type
// ---------------------------------------------------------------------------

impl traits::Encoder for TokenizerBackend {
    fn encode(&self, input: &str, add_special_tokens: bool) -> Result<Encoding> {
        match self {
            Self::HuggingFace(t) => t.encode(input, add_special_tokens),
            Self::Tiktoken(t) => t.encode(input, add_special_tokens),
            Self::Mock(t) => t.encode(input, add_special_tokens),
        }
    }

    fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
        match self {
            Self::HuggingFace(t) => t.encode_batch(inputs, add_special_tokens),
            Self::Tiktoken(t) => t.encode_batch(inputs, add_special_tokens),
            Self::Mock(t) => t.encode_batch(inputs, add_special_tokens),
        }
    }
}

impl traits::Decoder for TokenizerBackend {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String> {
        match self {
            Self::HuggingFace(t) => t.decode(token_ids, skip_special_tokens),
            Self::Tiktoken(t) => t.decode(token_ids, skip_special_tokens),
            Self::Mock(t) => t.decode(token_ids, skip_special_tokens),
        }
    }
}

impl traits::Tokenizer for TokenizerBackend {
    fn vocab_size(&self) -> usize {
        match self {
            Self::HuggingFace(t) => t.vocab_size(),
            Self::Tiktoken(t) => t.vocab_size(),
            Self::Mock(t) => t.vocab_size(),
        }
    }

    fn get_special_tokens(&self) -> &SpecialTokens {
        match self {
            Self::HuggingFace(t) => t.get_special_tokens(),
            Self::Tiktoken(t) => t.get_special_tokens(),
            Self::Mock(t) => t.get_special_tokens(),
        }
    }

    fn token_to_id(&self, token: &str) -> Option<TokenIdType> {
        match self {
            Self::HuggingFace(t) => t.token_to_id(token),
            Self::Tiktoken(t) => t.token_to_id(token),
            Self::Mock(t) => t.token_to_id(token),
        }
    }

    fn id_to_token(&self, id: TokenIdType) -> Option<String> {
        match self {
            Self::HuggingFace(t) => t.id_to_token(id),
            Self::Tiktoken(t) => t.id_to_token(id),
            Self::Mock(t) => t.id_to_token(id),
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        // Return self so callers can downcast to TokenizerBackend
        self
    }

    fn apply_chat_template(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
    ) -> Result<String> {
        match self {
            Self::HuggingFace(t) => t.apply_chat_template(messages, params),
            Self::Tiktoken(t) => t.apply_chat_template(messages, params),
            Self::Mock(t) => t.apply_chat_template(messages, params),
        }
    }

    fn chat_template_content_format(&self) -> ChatTemplateContentFormat {
        match self {
            Self::HuggingFace(t) => t.chat_template_content_format(),
            Self::Tiktoken(t) => t.chat_template_content_format(),
            Self::Mock(t) => t.chat_template_content_format(),
        }
    }

    fn set_chat_template(&mut self, template: String) -> Result<()> {
        match self {
            Self::HuggingFace(t) => t.set_chat_template(template),
            Self::Tiktoken(t) => t.set_chat_template(template),
            Self::Mock(t) => t.set_chat_template(template),
        }
    }
}

// ---------------------------------------------------------------------------
// Native incremental decode for the HuggingFace backend
// ---------------------------------------------------------------------------

impl TokenizerBackend {
    /// Incremental decode step — called once per generated token.
    ///
    /// For HuggingFace tokenizers, this delegates to the native `step_decode_stream`
    /// in the `tokenizers` crate which has optimized internal state management.
    /// For other backends, falls back to the standard double-decode algorithm.
    #[inline]
    pub fn decode_step(
        &self,
        token_id: TokenIdType,
        ids: &mut Vec<TokenIdType>,
        prefix: &mut String,
        prefix_index: &mut usize,
        skip_special_tokens: bool,
    ) -> Result<Option<String>> {
        match self {
            Self::HuggingFace(t) => t.decode_step_native(
                token_id,
                ids,
                prefix,
                prefix_index,
                skip_special_tokens,
            ),
            _ => self.decode_step_fallback(
                token_id,
                ids,
                prefix,
                prefix_index,
                skip_special_tokens,
            ),
        }
    }

    /// Fallback double-decode algorithm for non-HF backends.
    fn decode_step_fallback(
        &self,
        token_id: TokenIdType,
        ids: &mut Vec<TokenIdType>,
        prefix: &mut String,
        prefix_index: &mut usize,
        skip_special_tokens: bool,
    ) -> Result<Option<String>> {
        // Recompute prefix if it was cleared (first call or after incomplete UTF-8)
        if prefix.is_empty() && !ids.is_empty() {
            let new_prefix = self.decode(ids, skip_special_tokens)?;
            if !new_prefix.ends_with("�") {
                *prefix = new_prefix;
                *prefix_index = ids.len();
            }
        }

        ids.push(token_id);
        let string = self.decode(ids, skip_special_tokens)?;

        if string.len() > prefix.len() && !string.ends_with("�") {
            // Find char-safe split point
            let mut split_at = prefix.len();
            while !string.is_char_boundary(split_at) && split_at > 0 {
                split_at -= 1;
            }

            let new_text = string[split_at..].to_string().replace("�", "");

            // Drain consumed tokens and recompute prefix for next call
            let new_prefix_len = ids.len() - *prefix_index;
            ids.drain(..*prefix_index);
            *prefix_index = new_prefix_len;
            *prefix = self.decode(ids, skip_special_tokens)?;

            Ok(Some(new_text))
        } else {
            Ok(None)
        }
    }
}
