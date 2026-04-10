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

**Domain:** Machine Learning (CS229 Lecture Notes — Andrew Ng)

**Tại sao nhóm chọn domain này?**
> Tài liệu CS229 của Andrew Ng là nội dung học thuật có cấu trúc rõ ràng theo header (##, ###, ####),
> giúp kiểm chứng chiến lược chunking dựa trên ranh giới ngữ nghĩa. Domain ML có nhiều khái niệm kỹ thuật
> (gradient descent, logistic regression, SVM...) đòi hỏi chunk phải giữ đủ context để retrieval có nghĩa —
> đây là bài test tốt cho cả hai strategy.

### Data Inventory

| # | Tên tài liệu        | Nguồn                    | Số ký tự | Metadata đã gán                    |
|---|---------------------|--------------------------|----------|------------------------------------|
| 1 | machine_learning.md | CS229 Lecture Notes (Ng) | 35,774   | topic, section, subsection, source |

### Metadata Schema

| Trường metadata | Kiểu   | Ví dụ giá trị                | Tại sao hữu ích cho retrieval?                                           |
|-----------------|--------|------------------------------|--------------------------------------------------------------------------|
| `topic`         | string | `"CS229 Lecture notes"`      | Identify top-level document; useful if multiple docs are in same index   |
| `section`       | string | `"Supervised learning"`      | Filter by major section (e.g. only retrieve from Part II classification) |
| `subsection`    | string | `"Part I Linear Regression"` | Narrow to specific algorithm section                                     |
| `subsubsection` | string | `"1 LMS algorithm"`          | Most granular header — pinpoints exact algorithm discussion              |
| `source`        | string | `"machine_learning"`         | Identifies original file; supports multi-document indexes                |
| `strategy`      | string | `"markdown_header"`          | Tracks which chunking strategy produced this chunk (for A/B analysis)    |

---

## 3. Chunking Strategy - Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên `machine_learning.md` (chunk_size=200):

| Tài liệu            | Strategy                         | Chunk Count | Avg Length | Preserves Context?      |
|---------------------|----------------------------------|-------------|------------|-------------------------|
| machine_learning.md | FixedSizeChunker (`fixed_size`)  | 207         | 195 chars  | Partial — cắt giữa câu  |
| machine_learning.md | SentenceChunker (`by_sentences`) | 180         | 200 chars  | Tốt — giữ ranh giới câu |
| machine_learning.md | RecursiveChunker (`recursive`)   | 166         | 211 chars  | Tốt — ưu tiên \n\n      |

### Strategy Của Tôi (Phase 2)

**Loại:** LangChain `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` (custom strategy)

**Mô tả cách hoạt động:**
> Bước 1: `MarkdownHeaderTextSplitter` tách tài liệu tại các header `#`, `##`, `###`, `####`, lưu tên header
> vào metadata (`topic`, `section`, `subsection`, `subsubsection`). Mỗi section được giữ nguyên vẹn như một
> đơn vị ngữ nghĩa. Bước 2: `RecursiveCharacterTextSplitter` (chunk_size=500, overlap=50) sub-split những
> section dài, đảm bảo không có chunk nào vượt ngưỡng kích thước. Kết quả: 87 chunks trên machine_learning.md,
> avg length ~417 chars, mỗi chunk đều có metadata section để filter.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> CS229 có cấu trúc header rõ ràng: mỗi section (LMS algorithm, Normal equations, Logistic regression...)
> là một khái niệm ML độc lập. Chunking theo header giữ toàn bộ phần giải thích của một khái niệm trong cùng
> chunk, giúp retrieval luôn trả về context đầy đủ thay vì nửa chừng một công thức hay giải thích.

**Code snippet:**

```python
# src/langchain_chunking.py
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

chunker = LangChainMarkdownChunker(chunk_size=500, chunk_overlap=50)
docs = chunker.chunk(text, source="machine_learning")
# Returns Document objects with metadata: {topic, section, subsection, source, strategy}
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu            | Strategy                    | Chunk Count | Avg Length | Top-1 Score (Q1) |
|---------------------|-----------------------------|-------------|------------|------------------|
| machine_learning.md | RecursiveChunker (baseline) | 106         | 339 chars  | 0.801            |
| machine_learning.md | **Markdown Header (tôi)**   | 87          | 417 chars  | **0.809**        |

### So Sánh Với Thành Viên Khác

| Thành viên  | Strategy             | Avg Top-1 Score | Điểm mạnh                              | Điểm yếu                            |
|-------------|----------------------|-----------------|----------------------------------------|-------------------------------------|
| Tôi         | Markdown Header (LC) | 0.823           | Section metadata, higher avg score     | Ít chunks hơn, bỏ sót cross-section |
| (nhóm khác) | LC Recursive         | 0.811           | Granular chunks, overlap giúp boundary | Không có section metadata           |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Markdown Header chunking tốt hơn trên 4/5 queries vì CS229 có cấu trúc header học thuật rõ ràng — mỗi
> section tự nhiên là một đơn vị ngữ nghĩa hoàn chỉnh. Strategy B (LC Recursive) thắng ở query về LWLR vì
> câu trả lời liên quan nằm rải rác ở 3 đoạn khác nhau trong cùng section, nên chunk nhỏ hơn + overlap lại
> bắt được ranh giới tốt hơn trong trường hợp đó.

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

Dùng `LocalEmbedder` (all-MiniLM-L6-v2) + `compute_similarity()`:

| Pair | Sentence A                                           | Sentence B                                         | Dự đoán | Actual Score | Đúng?                           |
|------|------------------------------------------------------|----------------------------------------------------|---------|--------------|---------------------------------|
| 1    | "Gradient descent minimizes the cost function."      | "We update theta to reduce J(theta)."              | high    | 0.721        | Yes                             |
| 2    | "Logistic regression uses the sigmoid function."     | "The sigmoid maps real numbers to (0,1)."          | high    | 0.812        | Yes                             |
| 3    | "Linear regression fits a line to data."             | "The Amazon rainforest has millions of species."   | low     | 0.041        | Yes                             |
| 4    | "The normal equation solves for theta analytically." | "Gradient descent is an iterative optimization."   | medium  | 0.523        | Surprise — higher than expected |
| 5    | "Overfitting occurs when a model is too complex."    | "Regularization penalizes large parameter values." | high    | 0.674        | Yes                             |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Pair 4 bất ngờ nhất: "normal equation" và "gradient descent" đều là phương pháp tối ưu hóa cho linear
> regression, nên embedding model nhận ra chúng cùng semantic cluster (score=0.523 — medium-high) dù surface
> form hoàn toàn khác nhau. Điều này cho thấy embeddings biểu diễn khái niệm và chức năng, không chỉ từ
> khoá. Ngược lại, Pair 1 thấp hơn Pair 2 vì "gradient descent" + "cost function" là một phát biểu cụ thể
> hơn, trong khi Pair 2 (sigmoid definition) là quan hệ định nghĩa trực tiếp nên score cao hơn.

---

## 6. Results - Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với
các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query                                                                               | Gold Answer                                                                                       |
|---|-------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| 1 | What is gradient descent and how does it minimize the cost function?                | θ_j := θ_j − α * ∂J/∂θ_j, iteratively steps in direction of steepest decrease of J(θ)             |
| 2 | What is the cost function used in linear regression?                                | J(θ) = ½ Σ (h_θ(x^(i)) − y^(i))², the least-squares cost function                                 |
| 3 | How does logistic regression handle classification problems?                        | Uses sigmoid g(z)=1/(1+e^−z) to output P(y=1\|x); maximize log-likelihood via gradient ascent     |
| 4 | What is the normal equation and when is it used instead of gradient descent?        | θ = (X^T X)^−1 X^T y; used when n is small enough to invert X^T X analytically                    |
| 5 | What is locally weighted linear regression and how does it differ from standard LR? | LWLR weights training examples by w^(i)=exp(−(x^(i)−x)²/2τ²); non-parametric, re-trains per query |

### Kết Quả Của Tôi (Strategy A — Markdown Header, Pinecone + BAAI/bge-large-en-v1.5)

| # | Query                              | Top-1 Retrieved Chunk (tóm tắt)                                                           | Score | Relevant? | Agent Answer (tóm tắt)      |
|---|------------------------------------|-------------------------------------------------------------------------------------------|-------|-----------|-----------------------------|
| 1 | Gradient descent & cost function   | "...gradient descent on the original cost function J..." (section: Supervised learning)   | 0.809 | Yes       | LMS update rule explanation |
| 2 | Cost function in linear regression | "...how close the h(x^(i))'s are to y^(i)'s. We define..." (section: Supervised learning) | 0.828 | Yes       | J(θ)=½Σ(hθ−y)² definition   |
| 3 | Logistic regression classification | "Part II Classification and logistic regression..." (section: Supervised learning)        | 0.827 | Yes       | sigmoid function + MLE      |
| 4 | Normal equation                    | "Normal equations — Gradient descent gives one way..." (section: Supervised learning)     | 0.830 | Yes       | θ=(X^TX)^−1 X^Ty formula    |
| 5 | LWLR vs standard LR                | "...properties of the LWR algorithm..." (section: Supervised learning)                    | 0.819 | Yes       | Weighting + non-parametric  |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm - Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> So sánh trực tiếp kết quả A/B trên cùng bộ queries cho thấy chunking strategy ảnh hưởng đáng kể đến
> retrieval score — cùng embedder, cùng index nhưng strategy khác nhau có thể chênh 0.05–0.06 cosine score
> trên một query. Đặc biệt, thành viên dùng chunk nhỏ hơn (300 chars) cho top-3 đa dạng hơn, trong khi
> chunk lớn hơn của tôi cho top-1 score cao hơn nhưng top-3 ít phủ rộng hơn.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Nhóm khác dùng metadata `difficulty` (easy/medium/hard) để filter câu hỏi theo độ khó — rất hữu ích cho
> FAQ domain nhưng khó áp dụng cho tài liệu học thuật. Điều này nhắc nhở rằng metadata schema phải được
> thiết kế dựa trên query patterns thực tế, không phải chỉ dựa trên cấu trúc tài liệu.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**

**Failure case:** Query "What is the normal equation?" với strategy B (LC Recursive) trả về top-1 là
chunk chỉ chứa header `"#### 2 The normal equations"` (score=0.790) mà không có nội dung — chunk quá nhỏ
do split tại `\n\n` ngay sau header. Strategy A không gặp lỗi này vì MarkdownHeaderTextSplitter merge header
vào content của section. **Cải thiện:** Thêm `strip_headers=False` + min_chunk_size filter để loại bỏ
chunk chỉ có header không có body.

> Nếu làm lại, tôi sẽ tăng chunk_overlap từ 50 lên 100 để giảm mất thông tin tại ranh giới, và thêm
> trường metadata `algorithm` (ví dụ: "gradient_descent", "logistic_regression") để support filtered search
> theo thuật toán. Ngoài ra, tôi sẽ thử thêm một strategy kết hợp: dùng header để xác định section boundary,
> sau đó dùng sentence-level chunking bên trong mỗi section thay vì character-level.

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
