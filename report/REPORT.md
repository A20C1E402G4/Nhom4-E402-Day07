# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Hoàng Đinh Duy Anh
**Nhóm:** 4, E402
**Ngày:** 2026-04-10

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Hai vector có cosine similarity cao khi chúng chỉ gần cùng một hướng trong không gian embedding, tức là hai đoạn văn
> bản mang nội dung ngữ nghĩa tương đồng. Giá trị bằng 1.0 nghĩa là hai vector hoàn toàn trùng hướng; bằng 0.0 nghĩa là
> trực giao (không liên quan); bằng -1.0 nghĩa là ngược hướng hoàn toàn.

**Ví dụ HIGH similarity:**

- Sentence A: "Python is a high-level programming language used for software development."
- Sentence B: "Python is a popular language for writing software and scripts."
- Tại sao tương đồng: Cả hai câu đều nói về Python trong bối cảnh lập trình phần mềm - chủ đề, từ khoá, và ý định gần
  như giống nhau, khiến embedding model ánh xạ chúng về gần cùng một vùng trong không gian vector.

**Ví dụ LOW similarity:**

- Sentence A: "Python is a high-level programming language used for software development."
- Sentence B: "The Amazon rainforest is home to millions of plant and animal species."
- Tại sao khác: Hai câu thuộc hai domain hoàn toàn không liên quan (lập trình vs. sinh thái học nhiệt đới) - không có từ
  khoá hay khái niệm chung, nên vector của chúng chỉ theo những hướng gần như trực giao.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity đo góc giữa hai vector nên scale-invariant: một câu ngắn và một câu dài cùng chủ đề sẽ có embedding
> gần cùng hướng nhưng khác độ lớn, và cosine similarity vẫn cho kết quả cao trong khi Euclidean distance bị phóng đại
> bởi
> sự chênh lệch độ lớn đó. Với các embedding model được normalize về unit-norm, cosine similarity còn tương đương dot
> product - tính toán rất nhanh.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Phép tính:
> - `step = chunk_size − overlap = 500 − 50 = 450`
> - Các start positions: `0, 450, 900, …` - dừng khi `start + 500 ≥ 10 000`, tức `start ≥ 9 500`.
> - Start cuối hợp lệ: `9 900` (vì `9 900 < 10 000`; chunk này lấy `text[9900:10400]` tức phần còn lại).
> - Số chunks = `⌊9 900 / 450⌋ + 1 = 22 + 1 = 23`
>
> Đáp án: **23 chunks** (xác nhận bằng `FixedSizeChunker(chunk_size=500, overlap=50).chunk("a"*10000)`).

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Với overlap=100: `step = 400`, start cuối = `9 600`, số chunks = `9 600/400 + 1 = 25` - nhiều hơn 2 chunks so với
> overlap=50. Overlap lớn hơn giúp đảm bảo thông tin nằm gần ranh giới chunk luôn xuất hiện trong ít nhất hai chunk liền
> kề, giảm nguy cơ mất context khi retrieval chỉ trả về một chunk.

---

## 2. Document Selection - Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** [ví dụ: Customer support FAQ, Vietnamese law, cooking recipes, ...]

**Tại sao nhóm chọn domain này?**
> *Viết 2-3 câu:*

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 |              |       |          |                 |
| 2 |              |       |          |                 |
| 3 |              |       |          |                 |
| 4 |              |       |          |                 |
| 5 |              |       |          |                 |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|-----------------|------|---------------|--------------------------------|
|                 |      |               |                                |
|                 |      |               |                                |

---

## 3. Chunking Strategy - Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy                         | Chunk Count | Avg Length | Preserves Context? |
|----------|----------------------------------|-------------|------------|--------------------|
|          | FixedSizeChunker (`fixed_size`)  |             |            |                    |
|          | SentenceChunker (`by_sentences`) |             |            |                    |
|          | RecursiveChunker (`recursive`)   |             |            |                    |

### Strategy Của Tôi

**Loại:** [FixedSizeChunker / SentenceChunker / RecursiveChunker / custom strategy]

**Mô tả cách hoạt động:**
> *Viết 3-4 câu: strategy chunk thế nào? Dựa trên dấu hiệu gì?*

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Viết 2-3 câu: domain có pattern gì mà strategy khai thác?*

**Code snippet (nếu custom):**

```python
# Paste implementation here
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy      | Chunk Count | Avg Length | Retrieval Quality? |
|----------|---------------|-------------|------------|--------------------|
|          | best baseline |             |            |                    |
|          | **của tôi**   |             |            |                    |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|------------|----------|-----------------------|-----------|----------|
| Tôi        |          |                       |           |          |
| [Tên]      |          |                       |           |          |
| [Tên]      |          |                       |           |          |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:*

---

## 4. My Approach - Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** - approach:
> Dùng regex lookbehind `(?<=[.!?]) |(?<=\.)\n` để tách tại khoảng trắng sau dấu `.`, `!`, `?`, hoặc tại ký tự xuống
> dòng sau dấu chấm - lookbehind giữ nguyên dấu câu ở cuối câu trước thay vì loại bỏ nó. Sau khi tách, mỗi phần được
`.strip()` và lọc chuỗi rỗng, rồi các câu được nhóm liên tiếp theo `max_sentences_per_chunk` và nối bằng dấu cách.
>
> Edge cases được xử lý:
> - **Empty text**: guard `if not text: return []` ở đầu hàm.
> - **`max_sentences_per_chunk < 1`**: clamped về `1` trong `__init__` bằng `max(1, max_sentences_per_chunk)`, tránh
    `range(0, n, 0)` gây `ValueError`.
> - **Whitespace-only strings sau khi split**: lọc bằng `if s.strip()` để không tạo chunk rỗng.
> - **Text không có dấu câu**: regex không match → toàn bộ text là một "câu" duy nhất, được trả về làm một chunk.

**`RecursiveChunker.chunk` / `_split`** - approach:
> Algorithm hoạt động theo kiểu greedy-merge + đệ quy ưu tiên separator: thử tách text bằng separator ưu tiên cao nhất
> còn lại (`\n\n` → `\n` → `. ` → ` ` → `""`); với mỗi separator, ghép tham lam các mảnh lại miễn tổng không vượt
`chunk_size`, khi vượt thì flush chunk hiện tại và đệ quy xuống separator tiếp theo nếu mảnh đơn lẻ cũng quá lớn.
>
> Base cases:
> - `len(current_text) <= chunk_size` → trả luôn `[current_text]`, không cần tách tiếp.
> - `not remaining_separators` hoặc `separator == ""` → cắt theo ký tự (`text[i:i+chunk_size]`), đây là fallback cuối
    cùng đảm bảo thuật toán luôn kết thúc.
> - `not current_text` → trả `[]`, tránh xử lý chuỗi rỗng.

### EmbeddingStore

**`add_documents` + `search`** - approach:
> `add_documents` dùng `_make_record` để tạo dict chuẩn hóa gồm `id = "{doc.id}_{_next_index}"` (tránh trùng ID qua
> nhiều lần gọi), `content`, `embedding`, và `metadata` kèm thêm `doc_id`. Với backend in-memory, record được append vào
`self._store`; với ChromaDB, record được forward sang `collection.add()`.
>
> `search` tính similarity bằng dot product giữa query vector và embedding của từng record. Vì `MockEmbedder` (và
`LocalEmbedder` với `normalize_embeddings=True`) trả về unit-norm vector, dot product tương đương cosine similarity mà
> không cần chia thêm. Với ChromaDB, dùng `score = 1 - distance` để đổi L2 distance sang similarity. Edge case:
`min(top_k, collection.count())` tránh ChromaDB raise lỗi khi `top_k` vượt số doc đang lưu.

**`search_with_filter` + `delete_document`** - approach:
> `search_with_filter` áp dụng pattern **filter trước, rank sau**: nếu có `metadata_filter`, duyệt `self._store` để giữ
> lại chỉ các record có metadata khớp exact-match toàn bộ key-value trong filter dict, rồi mới chạy `_search_records`
> trên
> tập đã thu hẹp - đảm bảo mọi kết quả trả về đều thỏa điều kiện. Nếu `metadata_filter=None` thì đi thẳng vào
`search()`.
>
> `delete_document` xử lý theo từng backend: với ChromaDB gọi `collection.delete(where={"doc_id": doc_id})` trực tiếp;
> với in-memory, rebuild `self._store` bằng list comprehension loại bỏ mọi record có `metadata["doc_id"] == doc_id`. Cả
> hai branch so sánh count/len trước và sau để trả `True` nếu có ít nhất một record bị xóa, `False` nếu không tìm thấy
> doc.

### KnowledgeBaseAgent

**`answer`** - approach:
> `answer` thực hiện vòng lặp RAG ba bước: (1) gọi `store.search(question, top_k)` để lấy `top_k` chunk liên quan
> nhất, (2) nối `content` của các chunk bằng `"\n\n"` thành một context block, (3) dựng prompt theo format chuẩn rồi gọi
`llm_fn`.
>
> Prompt structure:
> ```
> Context:
> <chunk 1>
>
> <chunk 2>
> ...
>
> Question: <question>
> Answer:
> ```
> Suffix `Answer:` là convention phổ biến để steer instruction-following model bắt đầu sinh câu trả lời ngay tại đó.
> Context được inject bằng f-string đơn giản - không có post-processing trên output của `llm_fn`, giữ contract của
> method
> là string-in string-out thuần túy.

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.12.12, pytest-9.0.3, pluggy-1.6.0
rootdir: C:\Users\dduya\Work\project\2A202600064-HoangDinhDuyAnh-Day07
configfile: pyproject.toml
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================== 42 passed in 0.96s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions - Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán    | Actual Score | Đúng? |
|------|------------|------------|------------|--------------|-------|
| 1    |            |            | high / low |              |       |
| 2    |            |            | high / low |              |       |
| 3    |            |            | high / low |              |       |
| 4    |            |            | high / low |              |       |
| 5    |            |            | high / low |              |       |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:*

---

## 6. Results - Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với
các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 |       |             |
| 2 |       |             |
| 3 |       |             |
| 4 |       |             |
| 5 |       |             |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|---------------------------------|-------|-----------|------------------------|
| 1 |       |                                 |       |           |                        |
| 2 |       |                                 |       |           |                        |
| 3 |       |                                 |       |           |                        |
| 4 |       |                                 |       |           |                        |
| 5 |       |                                 |       |           |                        |

**Bao nhiêu queries trả về chunk relevant trong top-3?** __ / 5

---

## 7. What I Learned (5 điểm - Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:*

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá

| Tiêu chí                    | Loại    | Điểm tự đánh giá |
|-----------------------------|---------|------------------|
| Warm-up                     | Cá nhân | / 5              |
| Document selection          | Nhóm    | / 10             |
| Chunking strategy           | Nhóm    | / 15             |
| My approach                 | Cá nhân | / 10             |
| Similarity predictions      | Cá nhân | / 5              |
| Results                     | Cá nhân | / 10             |
| Core implementation (tests) | Cá nhân | / 30             |
| Demo                        | Nhóm    | / 5              |
| **Tổng**                    |         | **/ 100**        |
