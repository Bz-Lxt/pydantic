//! Tracking of validation errors back to positions in the original JSON input.
//!
//! Unlike the logical error location (`loc`), which records the schema path (field names and
//! indices) but not where the offending fragment occurs in the source, the structures here map
//! each error to a half-open **byte range** over the exact JSON bytes that were validated,
//! together with 1-based line/column coordinates for both endpoints.
//!
//! Coordinates are computed independently of [`jiter`] by a lightweight structural scan of the
//! same bytes, so they remain correct when:
//!
//! * fields are looked up through aliases (the logical path may use the field name rather than
//!   the key present in the JSON);
//! * the failing value comes from a particular union branch;
//! * an object contains duplicate keys (validation uses the last occurrence);
//! * the document contains multi-byte characters (offsets are byte offsets, columns count
//!   Unicode code points);
//! * the input is truncated and parsed with `allow_partial`.

use jiter::{JsonObject, JsonValue, PartialMode};

use crate::errors::{ErrorType, InputValue, LocItem, Location};

/// A half-open byte range `start..end` over the original JSON input, together with 1-based
/// line/column coordinates of both endpoints.
///
/// * `start_column`/`end_column` count Unicode code points from the start of the line, not bytes.
/// * for zero-width ranges (e.g. a missing field, or a syntax error) `start == end` and the
///   line/column pairs are equal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JsonPosition {
    /// Byte offset of the first byte of the fragment.
    pub start: usize,
    /// Byte offset one past the last byte of the fragment.
    pub end: usize,
    /// Line number of `start`, starting at 1.
    pub start_line: usize,
    /// Column number of `start`, starting at 1 (counts Unicode code points).
    pub start_column: usize,
    /// Line number of `end`, starting at 1.
    pub end_line: usize,
    /// Column number of `end`, starting at 1 (counts Unicode code points).
    pub end_column: usize,
}

/// Locate the fragment that a line error refers to in the original JSON document.
///
/// Returns `None` if the fragment cannot be identified; positions are best-effort metadata and
/// callers must always handle their absence.
pub(crate) fn locate(
    data: &[u8],
    allow_partial: PartialMode,
    root: &JsonValue<'_>,
    location: &Location,
    input_value: &InputValue,
    error_type: &ErrorType,
) -> Option<JsonPosition> {
    let tree = Scanner::scan(data, allow_partial)?;
    let loc: Vec<&LocItem> = match location {
        Location::List(items) => items.iter().rev().collect(),
        Location::Empty => Vec::new(),
    };
    let target = match input_value {
        InputValue::Json(value) => Some(value),
        // Python input values never correspond to the JSON document being scanned.
        InputValue::Python(_) => None,
    };
    let span = Resolver { data }.resolve(&tree, root, &loc, target, error_type)?;
    Some(span_to_position(data, span))
}

/// Build a zero-width position pointing at a single byte index, as reported by [`jiter`] for a
/// JSON syntax error. The index is clamped to the end of the (possibly truncated) input.
pub(crate) fn point_at(data: &[u8], mut index: usize) -> JsonPosition {
    if index > data.len() {
        index = data.len();
    }
    let line_col = LineIndex::build(data).line_col(data, index);
    JsonPosition {
        start: index,
        end: index,
        start_line: line_col.0,
        start_column: line_col.1,
        end_line: line_col.0,
        end_column: line_col.1,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Span {
    start: usize,
    end: usize,
}

impl Span {
    fn point(at: usize) -> Self {
        Self { start: at, end: at }
    }
}

#[derive(Debug)]
enum Node {
    Scalar,
    Array(Vec<NodeValue>),
    Object(Vec<Member>),
}

#[derive(Debug)]
struct NodeValue {
    span: Span,
    node: Node,
}

#[derive(Debug)]
struct Member {
    key_span: Span,
    key: String,
    value: NodeValue,
}

// ------------------------------------------------------------------------------------------------
// Scanning
// ------------------------------------------------------------------------------------------------

struct Scanner<'a> {
    data: &'a [u8],
    index: usize,
    /// `PartialMode::On` or `PartialMode::TrailingStrings` - jiter accepts truncated input.
    partial: bool,
    /// A truncated trailing string is included as a (partial) value.
    trailing_strings: bool,
}

impl<'a> Scanner<'a> {
    fn scan(data: &'a [u8], allow_partial: PartialMode) -> Option<NodeValue> {
        let mut scanner = Scanner {
            data,
            index: 0,
            partial: allow_partial.is_active(),
            trailing_strings: allow_partial.allow_trailing_str(),
        };
        scanner.whitespace();
        let value = scanner.value()?;
        if scanner.partial {
            // jiter does not check for trailing content in partial mode
            return Some(value);
        }
        scanner.whitespace();
        // only accept scans that consumed the whole document, like jiter's `finish()` does
        (scanner.index == data.len()).then_some(value)
    }

    fn whitespace(&mut self) {
        while matches!(self.data.get(self.index), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.index += 1;
        }
    }

    fn value(&mut self) -> Option<NodeValue> {
        self.whitespace();
        let start = self.index;
        let node = match self.data.get(self.index)? {
            b'{' => return self.object(start),
            b'[' => return self.array(start),
            b'"' => {
                self.string_raw()?;
                Node::Scalar
            }
            // true / false / null / NaN / Infinity and numbers all scan as a plain token.
            _ => {
                self.token();
                Node::Scalar
            }
        };
        Some(NodeValue {
            span: Span { start, end: self.index },
            node,
        })
    }

    fn array(&mut self, start: usize) -> Option<NodeValue> {
        self.index += 1; // [
        let mut items = Vec::new();
        self.whitespace();
        if self.data.get(self.index) == Some(&b']') {
            self.index += 1;
            return Some(NodeValue {
                span: Span { start, end: self.index },
                node: Node::Array(items),
            });
        }
        loop {
            match self.value() {
                Some(item) => items.push(item),
                // a truncated value in partial mode: the array itself ends at EOF
                None if self.partial => {
                    return Some(NodeValue {
                        span: Span {
                            start,
                            end: self.data.len(),
                        },
                        node: Node::Array(items),
                    });
                }
                None => return None,
            }
            self.whitespace();
            match self.data.get(self.index) {
                Some(b',') => {
                    self.index += 1;
                    self.whitespace();
                }
                Some(b']') => {
                    self.index += 1;
                    return Some(NodeValue {
                        span: Span { start, end: self.index },
                        node: Node::Array(items),
                    });
                }
                // jiter accepts EOF while expecting `,`/`]` in partial mode
                None if self.partial => {
                    return Some(NodeValue {
                        span: Span {
                            start,
                            end: self.data.len(),
                        },
                        node: Node::Array(items),
                    });
                }
                _ => return None,
            }
        }
    }

    fn object(&mut self, start: usize) -> Option<NodeValue> {
        self.index += 1; // {
        let mut members = Vec::new();
        self.whitespace();
        if self.data.get(self.index) == Some(&b'}') {
            self.index += 1;
            return Some(NodeValue {
                span: Span { start, end: self.index },
                node: Node::Object(members),
            });
        }
        loop {
            // key
            self.whitespace();
            let key_start = self.index;
            if self.data.get(self.index) != Some(&b'"') {
                // EOF (or anything else) with a partial document: object ends here, no more
                // members - matching jiter which returns the object parsed so far.
                return self.partial.then_some(NodeValue {
                    span: Span {
                        start,
                        end: self.data.len(),
                    },
                    node: Node::Object(members),
                });
            }
            let Some(raw) = self.string_raw() else {
                // unterminated key in a partial document - jiter keeps the members parsed so far
                return self.partial.then_some(NodeValue {
                    span: Span {
                        start,
                        end: self.data.len(),
                    },
                    node: Node::Object(members),
                });
            };
            let key_span = Span {
                start: key_start,
                end: self.index,
            };
            let key = decode_string(raw);
            self.whitespace();
            if self.data.get(self.index) != Some(&b':') {
                // truncated between key and value
                return self.partial.then_some(NodeValue {
                    span: Span {
                        start,
                        end: self.data.len(),
                    },
                    node: Node::Object(members),
                });
            }
            self.index += 1; // :
            let value = match self.value() {
                Some(value) => value,
                None if self.partial => {
                    // the member whose value is missing is dropped, matching jiter
                    return Some(NodeValue {
                        span: Span {
                            start,
                            end: self.data.len(),
                        },
                        node: Node::Object(members),
                    });
                }
                None => return None,
            };
            members.push(Member { key_span, key, value });
            self.whitespace();
            match self.data.get(self.index) {
                Some(b',') => {
                    self.index += 1;
                    self.whitespace();
                }
                Some(b'}') => {
                    self.index += 1;
                    return Some(NodeValue {
                        span: Span { start, end: self.index },
                        node: Node::Object(members),
                    });
                }
                None if self.partial => {
                    return Some(NodeValue {
                        span: Span {
                            start,
                            end: self.data.len(),
                        },
                        node: Node::Object(members),
                    });
                }
                _ => return None,
            }
        }
    }

    /// Consume a JSON string (positioned at the opening quote), returning the raw bytes
    /// **between** the quotes. Returns `None` on an unterminated string unless trailing partial
    /// strings are enabled.
    fn string_raw(&mut self) -> Option<&'a [u8]> {
        let start = self.index + 1;
        self.index += 1;
        loop {
            match self.data.get(self.index) {
                Some(b'"') => {
                    let raw = &self.data[start..self.index];
                    self.index += 1;
                    return Some(raw);
                }
                Some(b'\\') => {
                    // skip the escaped byte (or the four hex digits of a unicode escape)
                    self.index += 1;
                    if self.data.get(self.index) == Some(&b'u') {
                        self.index += 4;
                    }
                    self.index += 1;
                }
                Some(_) => self.index += 1,
                None => {
                    if self.trailing_strings {
                        // an escape at EOF may have advanced the index past the last byte;
                        // the partial span ends at EOF exactly
                        self.index = self.data.len();
                        return Some(&self.data[start..]);
                    }
                    return None;
                }
            }
        }
    }

    /// Consume a number or a `true`/`false`/`null`/`NaN`/`Infinity` literal token.
    fn token(&mut self) {
        while let Some(byte) = self.data.get(self.index) {
            if byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'+' | b'-') {
                self.index += 1;
            } else {
                break;
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Resolution of an error (logical path + offending value) to a scanned span
// ------------------------------------------------------------------------------------------------

struct Resolver<'a> {
    data: &'a [u8],
}

fn loc_key(item: &LocItem) -> Option<&str> {
    match item {
        LocItem::S(s) => Some(s.as_str()),
        LocItem::PyS(s) => Some(s),
        LocItem::I(_) => None,
    }
}

impl Resolver<'_> {
    fn resolve(
        &self,
        node: &NodeValue,
        value: &JsonValue<'_>,
        loc: &[&LocItem],
        target: Option<&JsonValue<'_>>,
        error_type: &ErrorType,
    ) -> Option<Span> {
        // No more path: this node is the fragment, provided it is the offending value.
        let Some((item, rest)) = loc.split_first() else {
            return if target.is_none_or(|t| t == value) {
                Some(node.span)
            } else {
                None
            };
        };

        match (&node.node, value, item) {
            (Node::Array(items), JsonValue::Array(array), LocItem::I(index)) => {
                let index = if *index >= 0 {
                    *index as usize
                } else {
                    // negative indices can appear in locations derived from `AliasPath`
                    array.len().checked_sub(index.unsigned_abs() as usize)?
                };
                match items.get(index).zip(array.get(index)) {
                    Some((child_node, child_value)) => self.resolve(child_node, child_value, rest, target, error_type),
                    // index past the end (e.g. a truncated array): point at the array end
                    None => Some(self.end_point(node.span)),
                }
            }
            (Node::Object(members), JsonValue::Object(object), item) => {
                let Some(key) = loc_key(item) else {
                    return contained(node, value, target);
                };
                self.resolve_object(members, object, key, rest, target, error_type, node.span)
            }
            // key/value kind mismatch (e.g. string loc through an array) - fall through to the
            // containment search below.
            _ => contained(node, value, target),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn resolve_object(
        &self,
        members: &[Member],
        object: &JsonObject<'_>,
        key: &str,
        rest: &[&LocItem],
        target: Option<&JsonValue<'_>>,
        error_type: &ErrorType,
        object_span: Span,
    ) -> Option<Span> {
        // direct path: validation uses the LAST occurrence of a duplicate key
        for (member, (_, member_value)) in members.iter().zip(object.iter()).rev() {
            if member.key == key
                && let Some(span) = self.resolve(&member.value, member_value, rest, target, error_type)
            {
                return Some(span);
            }
        }

        // object keys are validated too (e.g. `dict[str, constrained]`): the offending value is
        // then the key itself, not the member value
        if rest.is_empty()
            && let Some(JsonValue::Str(target_str)) = target
        {
            for member in members.iter().rev() {
                if member.key == target_str.as_ref() {
                    return Some(member.key_span);
                }
            }
        }

        // alias / `loc_by_alias=False`: the logical key is the field name while the JSON uses a
        // different key - search members (last first) for a subtree matching the remaining path
        // and offending value
        if target.is_some() {
            for (member, (_, member_value)) in members.iter().zip(object.iter()).rev() {
                if let Some(span) = self.resolve(&member.value, member_value, rest, target, error_type) {
                    return Some(span);
                }
            }
        }

        // key absent from the JSON object: a missing field points at where the next member would
        // go (the closing brace, or EOF for truncated input); anything else cannot be located
        if matches!(error_type, ErrorType::Missing { .. }) {
            Some(self.end_point(object_span))
        } else {
            None
        }
    }

    /// Position of a zero-width "insertion point" at the end of a container: just inside the
    /// closing bracket, or at EOF when the container was truncated.
    fn end_point(&self, span: Span) -> Span {
        let last = span.end.saturating_sub(1);
        if matches!(self.data.get(last), Some(b'}' | b']')) {
            Span::point(last)
        } else {
            Span::point(span.end)
        }
    }
}

/// Containment search used when the next location item cannot be applied directly (alias
/// mismatch against a non-object, or a schema path item that only exists lower down): search
/// every subtree (last occurrence first) for the offending value itself.
fn contained(node: &NodeValue, value: &JsonValue<'_>, target: Option<&JsonValue<'_>>) -> Option<Span> {
    // a scalar can only be the destination itself
    if target.is_some_and(|t| t == value) {
        return Some(node.span);
    }
    match (&node.node, value) {
        (Node::Array(items), JsonValue::Array(array)) => {
            for (child_node, child_value) in items.iter().zip(array.iter()).rev() {
                if let Some(span) = contained(child_node, child_value, target) {
                    return Some(span);
                }
            }
            None
        }
        (Node::Object(members), JsonValue::Object(object)) => {
            for (member, (_, member_value)) in members.iter().zip(object.iter()).rev() {
                if let Some(span) = contained(&member.value, member_value, target) {
                    return Some(span);
                }
            }
            None
        }
        _ => None,
    }
}

// ------------------------------------------------------------------------------------------------
// String decoding (only used to match object keys against logical paths)
// ------------------------------------------------------------------------------------------------

fn decode_string(raw: &[u8]) -> String {
    if !raw.contains(&b'\\') {
        // valid JSON means raw is valid UTF-8 here
        return String::from_utf8_lossy(raw).into_owned();
    }
    let mut out = String::with_capacity(raw.len());
    let mut i = 0;
    while i < raw.len() {
        match raw[i] {
            b'\\' => {
                i += 1;
                match raw.get(i) {
                    Some(b'"') => out.push('"'),
                    Some(b'/') => out.push('/'),
                    Some(b'b') => out.push('\u{0008}'),
                    Some(b'f') => out.push('\u{000C}'),
                    Some(b'n') => out.push('\n'),
                    Some(b'r') => out.push('\r'),
                    Some(b't') => out.push('\t'),
                    Some(b'u') => {
                        let (chr, len) = decode_unicode_escape(&raw[i + 1..]);
                        out.push(chr);
                        i += len;
                    }
                    // `\\` and any invalid escape (which jiter would already have rejected) keep
                    // the backslash
                    _ => out.push('\\'),
                }
                i += 1;
            }
            byte if byte < 0x80 => {
                out.push(byte as char);
                i += 1;
            }
            _ => {
                // copy the whole multi-byte UTF-8 sequence
                let start = i;
                i += 1;
                while i < raw.len() && raw[i] >= 0x80 {
                    i += 1;
                }
                out.push_str(&String::from_utf8_lossy(&raw[start..i]));
            }
        }
    }
    out
}

/// Decode a `uXXXX` escape at the start of `hex`, returning the character and the number of
/// additional bytes consumed beyond the `u` (4, or 10 for a surrogate pair).
fn decode_unicode_escape(hex: &[u8]) -> (char, usize) {
    let high = hex_value(hex, 0).unwrap_or(0xFFFD);
    if (0xD800..=0xDBFF).contains(&high) && hex.get(4) == Some(&b'\\') && hex.get(5) == Some(&b'u') {
        let low = hex_value(hex, 6).unwrap_or(0xFFFD);
        if (0xDC00..=0xDFFF).contains(&low) {
            let code = 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00);
            return (char::from_u32(code).unwrap_or('\u{FFFD}'), 10);
        }
    }
    (char::from_u32(high).unwrap_or('\u{FFFD}'), 4)
}

fn hex_value(hex: &[u8], offset: usize) -> Option<u32> {
    let mut value = 0u32;
    for i in offset..offset + 4 {
        let byte = *hex.get(i)?;
        let digit = match byte {
            b'0'..=b'9' => u32::from(byte - b'0'),
            b'a'..=b'f' => u32::from(byte - b'a' + 10),
            b'A'..=b'F' => u32::from(byte - b'A' + 10),
            _ => return None,
        };
        value = (value << 4) + digit;
    }
    Some(value)
}

// ------------------------------------------------------------------------------------------------
// Byte offsets -> line/column
// ------------------------------------------------------------------------------------------------

struct LineIndex {
    /// Byte offset at which each line starts, starting with 0.
    line_starts: Vec<usize>,
}

impl LineIndex {
    fn build(data: &[u8]) -> Self {
        let mut line_starts = vec![0usize];
        for (i, byte) in data.iter().enumerate() {
            if *byte == b'\n' {
                line_starts.push(i + 1);
            }
        }
        Self { line_starts }
    }

    /// 1-based `(line, column)` for a byte offset; columns count Unicode code points.
    fn line_col(&self, data: &[u8], offset: usize) -> (usize, usize) {
        let line = self.line_starts.partition_point(|&start| start <= offset) - 1;
        let line_start = self.line_starts[line];
        let column = match std::str::from_utf8(&data[line_start..offset.min(data.len())]) {
            Ok(slice) => slice.chars().count() + 1,
            // the input should always be valid UTF-8 after a successful parse, but never panic
            Err(_) => offset - line_start + 1,
        };
        (line + 1, column)
    }
}

fn span_to_position(data: &[u8], span: Span) -> JsonPosition {
    let index = LineIndex::build(data);
    let start = index.line_col(data, span.start);
    let end = index.line_col(data, span.end);
    JsonPosition {
        start: span.start,
        end: span.end,
        start_line: start.0,
        start_column: start.1,
        end_line: end.0,
        end_column: end.1,
    }
}
