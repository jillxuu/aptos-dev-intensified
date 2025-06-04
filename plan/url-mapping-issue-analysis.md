# URL Mapping Issue Analysis & Solution

## Problem Summary

The Aptos chatbot was generating incorrect URLs when answering questions, with URLs being truncated and missing specific page names.

### Examples of the Issue

**Incorrect URLs Generated:**
- `https://aptos.dev/en/build/guides#current-balance-for-a-coin` ❌
- `https://aptos.dev/en/build/guides#fungible-asset-balances` ❌
- `https://aptos.dev/en/build/indexer#build` ❌
- `https://aptos.dev/en/build/guides/exchanges#buildguides` ❌

**Correct URLs Expected:**
- `https://aptos.dev/en/build/guides/system-integrators-guide#current-balance-for-a-coin` ✅
- `https://aptos.dev/en/build/guides/exchanges#fungible-asset-balances` ✅
- `https://aptos.dev/en/build/indexer/indexer-api#proper-section` ✅

## Root Cause Analysis

### Initial Investigation

1. **Chunk Data Structure Issues**
   - Enhanced chunks stored malformed source paths: `en/data/developer-docs/apps/nextra/pages/en/build/guides/system-integrators-guide`
   - Should have been: `en/build/guides/system-integrators-guide`

2. **URL Lookup Failures**
   - Path registry couldn't find URLs for malformed paths
   - Returned `None` for URL lookups
   - LLM was left to generate its own URLs without proper context

3. **LLM URL Generation Issues**
   - When no valid URLs provided in context, LLM generated incorrect anchors
   - Created malformed anchors like `#build`, `#buildguides`, `#buildindexerindexer-api`
   - Used wrong section titles for anchor generation

### Deep Dive Analysis

**Chunk Data Problems:**
```json
{
  "source": "en/data/developer-docs/apps/nextra/pages/en/build/guides/system-integrators-guide",
  "section": "en/build/guides/system-integrators-guide", 
  "title": "",  // Often empty
  "content": "Context: Application Integration Guide > Application Integration Guide > ..."
}
```

**URL Mapping Registry:**
```yaml
en/build/guides/system-integrators-guide.mdx: en/build/guides/system-integrators-guide
en/build/guides/exchanges.mdx: en/build/guides/exchanges
```

**The Mismatch:**
- Chunks had: `en/data/developer-docs/apps/nextra/pages/en/build/guides/system-integrators-guide`
- Registry expected: `en/build/guides/system-integrators-guide`
- Result: URL lookup returned `None`

## Solution Options Considered

### Option 1: Fix at Chunk Generation Level (Recommended for Long-term)
**Approach:** Regenerate all enhanced chunks with correct source paths
**Pros:**
- Clean, permanent fix
- No runtime processing overhead
- Consistent data structure

**Cons:**
- Requires regenerating large chunk files
- Downtime during regeneration
- More complex deployment

### Option 2: Fix at Runtime During Context Formatting (Chosen)
**Approach:** Detect and correct malformed paths during LLM context preparation
**Pros:**
- Immediate fix without data regeneration
- No downtime required
- Backward compatible

**Cons:**
- Runtime processing overhead
- Band-aid solution
- Maintains inconsistent data

### Option 3: Fix in get_docs_url Function
**Approach:** Add fallback logic in URL generation function
**Pros:**
- Centralized fix location

**Cons:**
- Doesn't address root cause
- Would affect other parts of system
- Less targeted solution

## Implemented Solution

### Chosen Approach: Option 2 - Runtime Fix

**Location:** `app/routes/chat.py` in context formatting section

**Implementation Details:**

1. **Malformed Path Detection**
   ```python
   if "data/developer-docs/apps/nextra/pages/en/" in source_path:
   ```

2. **Path Correction Logic**
   ```python
   en_index = source_path.rfind("/en/")
   if en_index != -1:
       corrected_path = "en/" + source_path[en_index + 4:]
   ```

3. **URL Generation with Anchors**
   ```python
   chunk_url = path_registry.get_url(corrected_path)
   if chunk_url:
       full_url = f"{base_url}/{chunk_url.lstrip('/')}"
       # Add proper anchor generation
   ```

4. **Section Title Extraction**
   ```python
   # Extract from content pattern: "Context: ... > Section Title"
   if content.startswith('Context:'):
       if '>' in first_line:
           section_title = first_line.split('>')[-1].strip()
   ```

5. **Anchor Generation**
   ```python
   anchor = section_title.lower().replace(' ', '-').replace('(', '').replace(')', '')
   anchor = re.sub(r'[^a-z0-9\-]', '', anchor)
   anchor = re.sub(r'-+', '-', anchor).strip('-')
   ```

### Code Changes Made

**File:** `app/routes/chat.py`
**Lines:** ~673-695 and ~703-723

**Key Changes:**
1. Added malformed path detection and correction
2. Implemented proper section title extraction from content
3. Added robust anchor generation logic
4. Used `title` field instead of `section` field for anchor generation
5. Removed redundant URL instructions from prompt template

## Testing & Verification

### Test Results
**Before Fix:**
- URLs: `https://aptos.dev/en/build/guides#buildguides` ❌
- Missing specific page names
- Incorrect anchors

**After Fix:**
- URLs: `https://aptos.dev/en/build/guides/exchanges#fungible-asset-balances` ✅
- Correct page names included
- Proper section anchors

### Test Scripts Created
1. `debug_real_flow.py` - Full flow testing with OpenAI API
2. `test_url_fix_verification.py` - URL correction logic testing

## Issues Identified for Future Improvement

### 1. Template Complexity Impact
**Problem:** Complex production template reduces LLM comprehensiveness
**Evidence:** 
- Simple template: All 3 methods present (coin::balance, primary_fungible_store, GetFungibleAssetBalances)
- Production template: Only 2 methods present (missing GetFungibleAssetBalances)

**Potential Fix:** Simplify BASE_TEMPLATE to reduce cognitive load while maintaining essential instructions

### 2. Chunk Data Quality
**Problem:** Inconsistent data structure in enhanced chunks
**Issues:**
- Malformed source paths
- Empty title fields
- Inconsistent section naming

**Recommended Fix:** Regenerate chunks with proper data structure (Option 1 from above)

### 3. Template Redundancy
**Problem:** Redundant URL instructions in prompt template
**Fixed:** Removed duplicate URL handling instructions from section 7

## Future Recommendations

### Short-term (Current State)
- ✅ Runtime fix implemented and working
- ✅ Template redundancy removed
- 🔄 Monitor for performance impact of runtime processing

### Medium-term
- 📋 Simplify BASE_TEMPLATE to improve LLM comprehensiveness
- 📋 Add monitoring for URL generation accuracy
- 📋 Create automated tests for URL generation

### Long-term
- 📋 Regenerate enhanced chunks with correct source paths (Option 1)
- 📋 Implement proper data validation in chunk generation pipeline
- 📋 Add URL validation in chunk processing

## Technical Debt Notes

1. **Runtime Processing Overhead:** Current solution adds processing time to each LLM call
2. **Data Inconsistency:** Chunk data still contains malformed paths
3. **Maintenance Burden:** Runtime fix needs to be maintained alongside chunk generation changes

## Files Modified

1. `app/routes/chat.py` - Main implementation
2. `plan/url-mapping-issue-analysis.md` - This documentation

## Related Issues

- URL truncation in chatbot responses
- Incorrect anchor generation
- Template complexity affecting response comprehensiveness
- Chunk data quality issues

---

**Created:** 2025-06-04  
**Status:** Implemented (Runtime Fix), but not scalable to multiple data sources, and has also affected to the data source being used to answer a question which need more evaluation.  
**Next Review:** Consider chunk regeneration approach 