# Citing lesson docs: phases/11-llm-engineering/07-advanced-rag/docs/en.md
# Reference: Cormack et al., "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods" (2009)
# Reference: Robertson et al., "The Probabilistic Relevance Framework: BM25 and Beyond" (2009)
# Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE, 2022)

import math
from collections import Counter


def chunk_text(text, chunk_size=200, overlap=50):
    """
    基于单词数量的滑动窗口分块函数。
    
    参数:
        text (str): 待分块的原始文本。
        chunk_size (int): 每个分块（Chunk）的最大单词数。
        overlap (int): 相邻分块之间的重叠单词数，用于保持跨边界语义的连贯性。
        
    返回:
        list: 分割后的文本分块列表。
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        # 向后滑动窗口：移动距离为窗口大小减去重叠大小
        start += chunk_size - overlap
    return chunks


def build_vocabulary(documents):
    """
    基于文档语料库构建全局词汇表。
    
    参数:
        documents (list of str): 所有的文档/文本分块。
        
    返回:
        list: 按字母排序且不重复的词汇列表。
    """
    vocab = set()
    for doc in documents:
        # 将文本转换为小写并按空格切分，收集词项
        vocab.update(doc.lower().split())
    return sorted(vocab)


def compute_tf(text, vocab):
    """
    计算给定文本在全局词汇表上的词频（Term Frequency, TF）。
    
    参数:
        text (str): 输入文本。
        vocab (list): 全局词汇表。
        
    返回:
        list: 该文本对应的 TF 向量（每个词项在文本中的频率百分比）。
    """
    words = text.lower().split()
    count = Counter(words)
    total = len(words)
    if total == 0:
        return [0.0] * len(vocab)
    # 计算词频：该词在文本中出现的次数 / 文本总词数
    return [count.get(word, 0) / total for word in vocab]


def compute_idf(documents, vocab):
    """
    计算全局词汇表中每个词项的逆文档频率（Inverse Document Frequency, IDF）。
    
    参数:
        documents (list of str): 整个语料库中的所有文档分块。
        vocab (list): 全局词汇表。
        
    返回:
        list: 每个词项的 IDF 值列表。
    """
    n = len(documents)
    idf = []
    for word in vocab:
        # 统计有多少文档包含了当前词项
        doc_count = sum(1 for doc in documents if word in doc.lower().split())
        # 使用平滑的 IDF 公式，避免分母为 0 且防止出现负值
        idf.append(math.log((n + 1) / (doc_count + 1)) + 1)
    return idf


def tfidf_embed(text, vocab, idf):
    """
    生成给定文本的 TF-IDF 向量嵌入（Embedding）。
    
    参数:
        text (str): 输入文本。
        vocab (list): 全局词汇表。
        idf (list): 词汇表对应的 IDF 向量。
        
    返回:
        list: TF-IDF 嵌入向量。
    """
    tf = compute_tf(text, vocab)
    # 将每个词项的 TF 值乘以其全局 IDF 值
    return [t * i for t, i in zip(tf, idf)]


def cosine_similarity(a, b):
    """
    计算两个向量之间的余弦相似度（Cosine Similarity），用于评估它们的语义相关性。
    
    参数:
        a (list of float): 向量 A。
        b (list of float): 向量 B。
        
    返回:
        float: 相似度得分，范围在 [-1.0, 1.0] 之间。
    """
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def vector_search(query_embedding, stored_embeddings, top_k=5):
    """
    在已存向量中进行余弦相似度精确匹配的向量检索。
    
    参数:
        query_embedding (list): 查询文本的嵌入向量。
        stored_embeddings (list of list): 整个语料库分块的嵌入向量列表。
        top_k (int): 返回的最相似结果数。
        
    返回:
        list of tuple: 格式为 (doc_index, similarity_score) 的结果列表，按相似度降序排列。
    """
    scores = []
    for i, emb in enumerate(stored_embeddings):
        sim = cosine_similarity(query_embedding, emb)
        scores.append((i, sim))
    # 按照语义相似度从高到低排序
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


class BM25:
    """
    BM25 (Best Matching 25) 关键字检索算法的纯 Python 标准库实现。
    """
    def __init__(self, k1=1.2, b=0.75):
        """
        初始化 BM25 参数。
        
        参数:
            k1 (float): 词频饱和度控制参数。k1 越大，高频词的加分越不容易饱和。默认 1.2。
            b (float): 文档长度归一化惩罚程度。b=1 表示完全根据文档长度惩罚；b=0 表示关闭长度惩罚。默认 0.75。
        """
        self.k1 = k1
        self.b = b
        self.docs = []
        self.doc_lengths = []
        self.avg_dl = 0
        self.doc_freqs = {} # 词项 -> 包含该词项的文档数量 (DF)
        self.n_docs = 0

    def index(self, documents):
        """
        对输入文档库构建 BM25 倒排索引。
        
        参数:
            documents (list of str): 文档分块语料库。
        """
        self.docs = documents
        self.n_docs = len(documents)
        self.doc_lengths = []
        self.doc_freqs = {}

        for doc in documents:
            words = doc.lower().split()
            self.doc_lengths.append(len(words))
            # 统计文档频次（DF）
            unique_words = set(words)
            for word in unique_words:
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

        # 计算整个语料库的平均文档长度
        self.avg_dl = sum(self.doc_lengths) / self.n_docs if self.n_docs else 1

    def score(self, query, doc_idx):
        """
        计算特定查询对指定文档的 BM25 相关性得分。
        
        参数:
            query (str): 查询文本。
            doc_idx (int): 文档库中的文档索引。
            
        返回:
            float: BM25 相关性得分。
        """
        query_words = query.lower().split()
        doc_words = self.docs[doc_idx].lower().split()
        doc_len = self.doc_lengths[doc_idx]
        word_counts = Counter(doc_words)
        total = 0.0

        for term in query_words:
            if term not in word_counts:
                continue
            tf = word_counts[term]
            df = self.doc_freqs.get(term, 0)
            
            # 计算平滑的逆文档频率 (IDF)
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
            
            # 计算词频阻尼和文档长度惩罚项
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            
            # 累加查询中每个词项在目标文档中的得分贡献
            total += idf * numerator / denominator

        return total

    def search(self, query, top_k=10):
        """
        在索引中对查询进行关键字检索，返回得分最高的结果。
        
        参数:
            query (str): 查询文本。
            top_k (int): 返回的最大结果数量。
            
        返回:
            list of tuple: 格式为 (doc_index, bm25_score) 的结果列表，按得分降序排列。
        """
        scores = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def reciprocal_rank_fusion(ranked_lists, k=60):
    """
    倒数排名融合（Reciprocal Rank Fusion, RRF）算法。
    将多路异构检索器输出的排名列表（如 BM25 排名、向量语义排名）无缝融合成一个最终排名。
    
    参数:
        ranked_lists (list of list): 多个检索器的输出列表，每个列表内为 (doc_id, score) 元组。
        k (int): 平滑因子，控制排名前列的权重落差。默认值为 60。
        
    返回:
        list of tuple: 融合后的结果列表，格式为 (doc_id, rrf_score)，按 RRF 得分降序排列。
    """
    scores = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            # 核心公式: 1 / (k + rank + 1)，其中 rank 从 0 开始，所以加 1 转换成从 1 开始的真实排名
            scores[doc_id] += 1.0 / (k + rank + 1)
    # 按倒数排名融合的综合得分降序排序
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused


def hybrid_search(query, chunks, vector_embeddings, vocab, idf, bm25_index, top_k=5, retrieval_pool=15):
    """
    混合检索（Hybrid Search）：向量检索 + 关键字检索 + RRF 融合。
    
    参数:
        query (str): 查询文本。
        chunks (list): 文本分块列表。
        vector_embeddings (list): 文本分块对应的向量嵌入列表。
        vocab (list): 全局词表。
        idf (list): 全局 IDF 向量。
        bm25_index (BM25): 已经构建完毕的 BM25 索引实例。
        top_k (int): 最终合并列表需要返回的结果数量。
        retrieval_pool (int): 每一路子检索器检索候选集的池大小。
        
    返回:
        list of tuple: 混合检索到的最相似文档元组列表 (doc_id, rrf_score)。
    """
    # 1. 向量搜索获取 top候选集
    query_emb = tfidf_embed(query, vocab, idf)
    vec_results = vector_search(query_emb, vector_embeddings, top_k=retrieval_pool)
    
    # 2. BM25 关键字搜索获取 top候选集
    bm25_results = bm25_index.search(query, top_k=retrieval_pool)
    
    # 3. 使用 RRF 算法进行排名融合
    fused = reciprocal_rank_fusion([vec_results, bm25_results])
    return fused[:top_k]


def rerank(query, candidates, chunks):
    """
    轻量级重排序器（Reranker），启发式模拟 Cross-Encoder 的精细打分过程。
    结合了关键词交集（词项重合度）、二元组（Bi-gram）匹配度、首段权重提升和初筛基础分。
    
    参数:
        query (str): 用户查询。
        candidates (list): 初始检索候选文档的元组列表 (doc_id, initial_score)。
        chunks (list of str): 文档库文本分块。
        
    返回:
        list of tuple: 重排打分后的元组列表 (doc_id, rerank_score)，降序排列。
    """
    query_words = set(query.lower().split())
    # 定义标准虚词/停用词表，剔除噪音
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how",
                  "why", "when", "where", "do", "does", "for", "of", "in", "to",
                  "and", "or", "on", "at", "by", "it", "its", "this", "that",
                  "with", "from", "be", "has", "have", "had", "not", "but"}
    query_terms = query_words - stop_words

    scored = []
    for doc_id, initial_score in candidates:
        chunk = chunks[doc_id].lower()
        chunk_words = set(chunk.split())

        # Heuristic 1: 单词重合度（交集大小）
        term_overlap = len(query_terms & chunk_words)

        # Heuristic 2: 二元词组（Bi-gram）的精确子串匹配（考察局部语序连续性）
        query_bigrams = set()
        q_list = [w for w in query.lower().split() if w not in stop_words]
        for i in range(len(q_list) - 1):
            query_bigrams.add(q_list[i] + " " + q_list[i + 1])
        bigram_matches = sum(1 for bg in query_bigrams if bg in chunk)

        # Heuristic 3: 位置偏置（主旨一般在文本前 1/3，如果在前部命中关键词则给予加分）
        position_boost = 0
        for term in query_terms:
            pos = chunk.find(term)
            if pos != -1 and pos < len(chunk) // 3:
                position_boost += 0.5

        # 综合打分公式：结合初筛分与多种文本细粒度重合指标
        rerank_score = (
            term_overlap * 1.0
            + bigram_matches * 2.0
            + position_boost
            + initial_score * 5.0
        )
        scored.append((doc_id, rerank_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def hyde_generate_hypothesis(query):
    """
    HyDE（Hypothetical Document Embeddings，假设文档嵌入）的规则生成器。
    模拟大模型根据查询意图自动生成“伪文档”草稿，帮助缓和查询与真实文本之间的语义鸿沟。
    
    参数:
        query (str): 查询文本。
        
    返回:
        str: 自动生成的虚构/假设相关文档。
    """
    # 模拟针对不同疑问词的伪回答生成模板
    templates = {
        "what": "The answer to '{query}' is as follows: Based on our documentation, {topic} involves specific policies and procedures that define the process and requirements.",
        "how": "To address '{query}': The process involves several steps. First, you need to initiate the request for {topic}. Then, the system processes it according to the defined rules and policies.",
        "default": "Regarding '{query}': Our records indicate specific details and policies related to {topic} that provide a comprehensive answer to this question."
    }
    query_lower = query.lower().strip()
    if query_lower.startswith("what"):
        template = templates["what"]
    elif query_lower.startswith("how"):
        template = templates["how"]
    else:
        template = templates["default"]

    # 提取查询核心主题词，过滤常见连词
    filler = {"what", "is", "the", "how", "do", "does", "a", "an", "for", "of",
              "to", "in", "on", "at", "by", "and", "or", "are", "was", "were", "?"}
    topic_words = [w.strip("?.,!") for w in query.lower().split() if w.strip("?.,!") not in filler]
    topic = " ".join(topic_words) if topic_words else "this topic"

    return template.format(query=query, topic=topic)


def hyde_search(query, vector_embeddings, vocab, idf, top_k=5):
    """
    执行 HyDE 检索：利用生成的“假设伪文档”进行向量相似度匹配。
    
    参数:
        query (str): 原始查询。
        vector_embeddings (list): 全局候选文档向量。
        vocab (list): 全局词汇表。
        idf (list): 词汇表对应 IDF 向量。
        top_k (int): 返回最接近的结果数。
        
    返回:
        tuple: (检索到的元组结果列表, 生成的假设伪文档文本)
    """
    # 1. 生成假设性的相关伪文档（Hypothesis）
    hypothesis = hyde_generate_hypothesis(query)
    # 2. 将伪文档编码成语义向量
    hypothesis_emb = tfidf_embed(hypothesis, vocab, idf)
    # 3. 使用伪文档向量对真实文本库进行匹配，返回最接近真实文档
    results = vector_search(hypothesis_emb, vector_embeddings, top_k)
    return results, hypothesis


def create_parent_child_chunks(text, parent_size=200, child_size=50):
    """
    父子分块（Parent-Child Chunking）策略生成器。
    将大段文本拆分成大粒度的父文档和小粒度的子文档。检索时对子文档进行高精度语义匹配，
    匹配成功后将更完整的父文档内容提供给 LLM 作为上下文，解决“精准检索 vs 丰富上下文”的矛盾。
    
    参数:
        text (str): 待处理的完整原始长文本。
        parent_size (int): 父分块的最大单词数。
        child_size (int): 子分块的最大单词数。
        
    返回:
        tuple: (父分块列表, 子分块列表, 子分块索引到对应父分块索引的映射 map)
    """
    words = text.split()
    parents = []
    children = []
    child_to_parent = {}

    parent_idx = 0
    start = 0
    while start < len(words):
        parent_end = min(start + parent_size, len(words))
        parent_text = " ".join(words[start:parent_end])
        parents.append(parent_text)

        # 在当前父分块的文字范围内，无重叠地切分成子分块
        child_start = start
        while child_start < parent_end:
            child_end = min(child_start + child_size, parent_end)
            child_text = " ".join(words[child_start:child_end])
            child_idx = len(children)
            children.append(child_text)
            # 记录当前子文档所属的父文档 ID
            child_to_parent[child_idx] = parent_idx
            child_start += child_size

        parent_idx += 1
        start += parent_size

    return parents, children, child_to_parent


def evaluate_faithfulness(answer, retrieved_chunks):
    """
    基于启发式词重合度计算的 RAG 忠实度（Faithfulness）评估。
    检查模型输出的每一句话，是否都能在检索出的原始 Context 中找到支撑，从而识别并量化幻觉。
    
    参数:
        answer (str): 模型生成的回答。
        retrieved_chunks (list of str): 检索出来的参考上下文 chunks。
        
    返回:
        tuple: (忠实度得分 0.0~1.0, 缺乏事实依据的幻觉/无依据句子列表)
    """
    # 按句号分割成句子，并剔除过短的无效空句
    answer_sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not answer_sentences:
        return 1.0, []

    grounded = 0
    ungrounded = []
    context = " ".join(retrieved_chunks).lower()

    for sentence in answer_sentences:
        words = set(sentence.lower().split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or",
                      "to", "of", "in", "for", "on", "at", "by", "it", "this", "that"}
        content_words = words - stop_words
        if not content_words:
            grounded += 1
            continue

        # 检查句子里的核心实词，有多少在参考上下文中命中
        matched = sum(1 for w in content_words if w in context)
        ratio = matched / len(content_words) if content_words else 0

        # 如果至少一半的实词都在上下文中出现，认定该句子在上下文中有事实依据
        if ratio >= 0.5:
            grounded += 1
        else:
            ungrounded.append(sentence)

    # 忠实度得分 = 有依据句子数 / 总句子数
    score = grounded / len(answer_sentences) if answer_sentences else 1.0
    return score, ungrounded


def evaluate_retrieval_recall(queries_with_relevant, retrieval_fn, k=5):
    """
    评估检索系统在测试集上的平均召回率（Recall@K）。
    
    参数:
        queries_with_relevant (list): 格式为 (query, list_of_ground_truth_indices) 的测试元组。
        retrieval_fn (callable): 检索函数，接收 (query, k) 并返回检索结果的 ID。
        k (int): 检索召回的 TopK 截止位置。
        
    返回:
        tuple: (平均召回率 0.0~1.0, 每条 query 的详细召回指标列表)
    """
    total_recall = 0.0
    results = []

    for query, relevant_indices in queries_with_relevant:
        retrieved = retrieval_fn(query, k)
        retrieved_indices = set(idx for idx, _ in retrieved)
        relevant_set = set(relevant_indices)
        
        # 命中交集数
        hits = len(retrieved_indices & relevant_set)
        # 召回率 = 检索命中的真实数 / 所有真实数
        recall = hits / len(relevant_set) if relevant_set else 1.0
        total_recall += recall
        results.append({
            "query": query,
            "recall": recall,
            "hits": hits,
            "total_relevant": len(relevant_set)
        })

    avg_recall = total_recall / len(queries_with_relevant) if queries_with_relevant else 0
    return avg_recall, results


def build_rag_prompt(query, retrieved_chunks):
    """
    构建最终输入给 LLM 的系统 RAG 提示词（Prompt）。
    
    参数:
        query (str): 用户提问。
        retrieved_chunks (list of str): 检索获取的相关知识上下文。
        
    返回:
        str: 拼接完成的 RAG 提示词。
    """
    # 格式化上下文块，标明来源序号
    context = "\n\n---\n\n".join(
        f"[Source {i+1}]\n{chunk}"
        for i, chunk in enumerate(retrieved_chunks)
    )
    return (
        "Answer the question based ONLY on the following context.\n"
        "If the context doesn't contain enough information, "
        "say \"I don't have enough information to answer that.\"\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


# 虚拟的 Acme 公司业务与政策数据库作为测试文档集
SAMPLE_DOCUMENTS = [
    """Acme Corp Refund Policy.
    All standard plan customers are eligible for a full refund within 30 days of purchase.
    Enterprise plan customers receive an extended 60-day refund window with pro-rated refunds
    calculated from the date of cancellation. Refunds are processed within 5-7 business days
    and returned to the original payment method. No refunds are available after the refund
    window closes. Customers must submit refund requests through the support portal or by
    contacting their account manager directly. Annual subscriptions that are cancelled mid-term
    will receive a pro-rated credit for the remaining months.""",

    """Acme Corp Product Overview.
    Acme Corp offers three product tiers: Starter, Professional, and Enterprise.
    The Starter plan includes basic features for individual users at $29 per month.
    The Professional plan adds team collaboration, advanced analytics, and priority
    support for $99 per month per user. The Enterprise plan includes everything in
    Professional plus custom integrations, dedicated account management, SSO,
    audit logs, and a 99.99% uptime SLA. Enterprise pricing is custom and starts
    at $500 per month for up to 50 users. All plans include a 14-day free trial
    with no credit card required.""",

    """Acme Corp Security Practices.
    Acme Corp maintains SOC 2 Type II compliance and undergoes annual third-party
    security audits. All data is encrypted at rest using AES-256 and in transit
    using TLS 1.3. Customer data is stored in isolated tenants within AWS
    us-east-1 and eu-west-1 regions. Data residency can be configured per
    organization for Enterprise customers. Backups are performed every 6 hours
    with 30-day retention. Acme Corp does not sell or share customer data with
    third parties. Enterprise customers can request data deletion within 24 hours.
    Bug bounty program available through HackerOne.""",

    """Acme Corp API Documentation.
    The Acme API uses REST with JSON request and response bodies. Authentication
    is via Bearer tokens issued through OAuth 2.0. Rate limits are 100 requests
    per minute for Starter, 1000 for Professional, and 10000 for Enterprise.
    Rate limit headers are included in every response: X-RateLimit-Limit,
    X-RateLimit-Remaining, and X-RateLimit-Reset. Exceeding the rate limit
    returns HTTP 429 with a Retry-After header. The API supports pagination
    via cursor-based pagination using the next_cursor field. Webhooks are
    available for real-time event notifications on Professional and Enterprise
    plans. API versioning uses date-based versions in the URL path.""",

    """Acme Corp Q3 2025 Earnings Report.
    Total revenue for Q3 2025 was $47.2 million, up 23% year-over-year.
    Enterprise segment contributed $31.8 million, representing 67% of total
    revenue. Professional segment added $12.1 million. Starter segment
    contributed $3.3 million. Customer count grew to 14,200 from 11,800
    in Q3 2024. Net retention rate was 118%. Operating expenses were
    $38.4 million. EBITDA was $8.8 million with an 18.6% margin.
    Free cash flow was $6.2 million. Guidance for Q4 2025 is $51-53 million
    in revenue with continued margin expansion.""",

    """Acme Corp Uptime and Reliability.
    Acme Corp guarantees 99.9% uptime for Professional plans and 99.99% uptime
    for Enterprise plans. Uptime is calculated monthly excluding scheduled
    maintenance windows which are announced 72 hours in advance. If uptime
    falls below the guaranteed level, customers receive service credits:
    10% credit for each 0.1% below the SLA threshold, up to a maximum of
    30% of the monthly fee. Service credits must be requested within 30 days
    of the incident. Status page updates are posted at status.acme.com
    within 5 minutes of any detected incident. Post-incident reports are
    published within 48 hours for any outage exceeding 15 minutes."""
]


if __name__ == "__main__":
    print("=" * 65)
    print("STEP 1: BM25 Keyword Search")
    print("=" * 65)

    # 对文档集进行细粒度预分块
    all_chunks = []
    chunk_sources = []
    source_names = ["refund", "product", "security", "api", "earnings", "uptime"]
    for i, doc in enumerate(SAMPLE_DOCUMENTS):
        doc_chunks = chunk_text(doc, chunk_size=50, overlap=10)
        for c in doc_chunks:
            all_chunks.append(c)
            chunk_sources.append(source_names[i])

    # 初始化 BM25 并编入词汇索引
    bm25 = BM25()
    bm25.index(all_chunks)

    test_query = "What was revenue last quarter?"
    bm25_results = bm25.search(test_query, top_k=5)
    print(f"  Query: {test_query}")
    print(f"  BM25 top-5:")
    for rank, (idx, score) in enumerate(bm25_results):
        preview = all_chunks[idx][:70].replace("\n", " ")
        print(f"    #{rank+1} [{chunk_sources[idx]}] score={score:.4f} | {preview}...")

    print("\n" + "=" * 65)
    print("STEP 2: Vector Search vs BM25")
    print("=" * 65)

    # 对全分块提取向量嵌入词袋表示（模拟 Vector DB 嵌入存储）
    vocab = build_vocabulary(all_chunks)
    idf = compute_idf(all_chunks, vocab)
    embeddings = [tfidf_embed(c, vocab, idf) for c in all_chunks]

    queries = [
        "What is the refund policy for enterprise customers?",
        "What was revenue last quarter?",
        "How is data encrypted?",
        "What are the API rate limits for enterprise?",
        "What happens if uptime falls below SLA?"
    ]

    # 对比同一组查询下，TF-IDF 向量搜索与 BM25 谁的 Top1 能准确抓到对应主题
    for query in queries:
        query_emb = tfidf_embed(query, vocab, idf)
        vec_top1 = vector_search(query_emb, embeddings, top_k=1)[0]
        bm25_top1 = bm25.search(query, top_k=1)[0]

        print(f"\n  Query: {query}")
        print(f"    Vector #1: [{chunk_sources[vec_top1[0]]}] score={vec_top1[1]:.4f}")
        print(f"    BM25   #1: [{chunk_sources[bm25_top1[0]]}] score={bm25_top1[1]:.4f}")
        agree = "AGREE" if chunk_sources[vec_top1[0]] == chunk_sources[bm25_top1[0]] else "DISAGREE"
        print(f"    {agree}")

    print("\n" + "=" * 65)
    print("STEP 3: Reciprocal Rank Fusion (Hybrid Search)")
    print("=" * 65)

    query = "What was revenue last quarter?"
    print(f"  Query: {query}")

    query_emb = tfidf_embed(query, vocab, idf)
    vec_results = vector_search(query_emb, embeddings, top_k=10)
    bm25_results = bm25.search(query, top_k=10)

    print(f"\n  Vector top-3:")
    for rank, (idx, score) in enumerate(vec_results[:3]):
        print(f"    #{rank+1} [{chunk_sources[idx]}] {score:.4f}")

    print(f"\n  BM25 top-3:")
    for rank, (idx, score) in enumerate(bm25_results[:3]):
        print(f"    #{rank+1} [{chunk_sources[idx]}] {score:.4f}")

    # 将上面独立出的向量和关键字列表排名进行倒数排名融合（RRF）
    fused = reciprocal_rank_fusion([vec_results, bm25_results])
    print(f"\n  RRF fused top-5:")
    for rank, (idx, score) in enumerate(fused[:5]):
        preview = all_chunks[idx][:60].replace("\n", " ")
        print(f"    #{rank+1} [{chunk_sources[idx]}] rrf={score:.4f} | {preview}...")

    print("\n" + "=" * 65)
    print("STEP 4: Reranking")
    print("=" * 65)

    query = "enterprise refund policy"
    print(f"  Query: {query}")

    # 1. 混合搜索出前 10 个候选块作为底排池
    hybrid_results = hybrid_search(query, all_chunks, embeddings, vocab, idf, bm25, top_k=10)
    # 2. 传入候选块和查询到细粒度重排序器中进行二次精排
    reranked = rerank(query, hybrid_results, all_chunks)

    print(f"\n  Before reranking (top-5):")
    for rank, (idx, score) in enumerate(hybrid_results[:5]):
        preview = all_chunks[idx][:60].replace("\n", " ")
        print(f"    #{rank+1} [{chunk_sources[idx]}] score={score:.4f} | {preview}...")

    print(f"\n  After reranking (top-5):")
    for rank, (idx, score) in enumerate(reranked[:5]):
        preview = all_chunks[idx][:60].replace("\n", " ")
        print(f"    #{rank+1} [{chunk_sources[idx]}] score={score:.4f} | {preview}...")

    print("\n" + "=" * 65)
    print("STEP 5: HyDE (Hypothetical Document Embeddings)")
    print("=" * 65)

    query = "How much money did the company make?"
    print(f"  Query: {query}")
    print(f"  (Note: query uses 'money', docs use 'revenue' and 'earnings')")

    # 对比直接用语义向量检索 vs. 通过 HyDE 生成假设文档后再检索的效果
    query_emb = tfidf_embed(query, vocab, idf)
    direct_results = vector_search(query_emb, embeddings, top_k=3)
    hyde_results, hypothesis = hyde_search(query, embeddings, vocab, idf, top_k=3)

    print(f"\n  Hypothesis: {hypothesis[:100]}...")

    print(f"\n  Direct search top-3:")
    for rank, (idx, score) in enumerate(direct_results):
        print(f"    #{rank+1} [{chunk_sources[idx]}] {score:.4f}")

    print(f"\n  HyDE search top-3:")
    for rank, (idx, score) in enumerate(hyde_results):
        print(f"    #{rank+1} [{chunk_sources[idx]}] {score:.4f}")

    print("\n" + "=" * 65)
    print("STEP 6: Parent-Child Chunking")
    print("=" * 65)

    # 合并所有文档，重新切分为父文档与子文档结构
    full_text = " ".join(SAMPLE_DOCUMENTS)
    parents, children, child_to_parent = create_parent_child_chunks(
        full_text, parent_size=100, child_size=25
    )

    print(f"  Total words: {len(full_text.split())}")
    print(f"  Parent chunks: {len(parents)} (100 words each)")
    print(f"  Child chunks: {len(children)} (25 words each)")
    print(f"  Ratio: {len(children)/len(parents):.1f} children per parent")

    child_vocab = build_vocabulary(children)
    child_idf = compute_idf(children, child_vocab)
    child_embeddings = [tfidf_embed(c, child_vocab, child_idf) for c in children]

    query = "enterprise refund 60 days"
    # 用小范围高灵敏的子文档进行嵌入和相似度匹配
    query_emb = tfidf_embed(query, child_vocab, child_idf)
    child_results = vector_search(query_emb, child_embeddings, top_k=3)

    print(f"\n  Query: {query}")
    print(f"\n  Matched children:")
    for rank, (idx, score) in enumerate(child_results):
        parent_idx = child_to_parent[idx]
        print(f"    Child #{idx} (score={score:.4f}):")
        print(f"      Child text: {children[idx][:80]}...")
        # 匹配到子文档后，给大模型返回其更大、包含完整上下文的父文档内容
        print(f"      Parent #{parent_idx}: {parents[parent_idx][:80]}...")

    print("\n" + "=" * 65)
    print("STEP 7: Faithfulness Evaluation")
    print("=" * 65)

    # 真实的回答（所有断言都在 Context 中有支持）
    good_answer = (
        "Enterprise customers receive a 60-day refund window. "
        "Refunds are pro-rated from the date of cancellation. "
        "Processing takes 5-7 business days."
    )
    # 虚假的回答（包含无事实根据的幻觉宣称，如 90 天，立刻处理，50刀服务费）
    bad_answer = (
        "Enterprise customers receive a 90-day refund window. "
        "Refunds are processed instantly. "
        "There is a $50 processing fee."
    )
    context_chunks = [all_chunks[i] for i, _ in hybrid_search(
        "enterprise refund", all_chunks, embeddings, vocab, idf, bm25, top_k=3
    )]

    good_score, good_ungrounded = evaluate_faithfulness(good_answer, context_chunks)
    bad_score, bad_ungrounded = evaluate_faithfulness(bad_answer, context_chunks)

    print(f"  Context: {len(context_chunks)} chunks about refund policy")
    print(f"\n  Good answer: \"{good_answer[:80]}...\"")
    print(f"  Faithfulness: {good_score:.2f}")
    if good_ungrounded:
        print(f"  Ungrounded claims: {good_ungrounded}")
    else:
        print(f"  All claims grounded in context.")

    print(f"\n  Bad answer: \"{bad_answer[:80]}...\"")
    print(f"  Faithfulness: {bad_score:.2f}")
    if bad_ungrounded:
        print(f"  Ungrounded claims:")
        for claim in bad_ungrounded:
            print(f"    - \"{claim}\"")

    print("\n" + "=" * 65)
    print("STEP 8: Full Advanced RAG Pipeline Comparison")
    print("=" * 65)

    # 对不同的单独检索与组合检索方式进行整体精度/命中召回比对
    comparison_queries = [
        ("What is the refund policy for enterprise?", "refund"),
        ("What was Q3 revenue?", "earnings"),
        ("How is customer data encrypted?", "security"),
        ("What are the API rate limits?", "api"),
        ("What is the uptime guarantee?", "uptime"),
    ]

    print(f"  {'Query':<45s} {'Vector':>8s} {'BM25':>8s} {'Hybrid':>8s} {'Rerank':>8s}")
    print("  " + "-" * 77)

    for query, expected_source in comparison_queries:
        query_emb = tfidf_embed(query, vocab, idf)

        vec_top = vector_search(query_emb, embeddings, top_k=1)[0]
        vec_hit = "HIT" if chunk_sources[vec_top[0]] == expected_source else "miss"

        bm25_top = bm25.search(query, top_k=1)[0]
        bm25_hit = "HIT" if chunk_sources[bm25_top[0]] == expected_source else "miss"

        hybrid_top = hybrid_search(query, all_chunks, embeddings, vocab, idf, bm25, top_k=1)[0]
        hybrid_hit = "HIT" if chunk_sources[hybrid_top[0]] == expected_source else "miss"

        hybrid_pool = hybrid_search(query, all_chunks, embeddings, vocab, idf, bm25, top_k=10)
        reranked_top = rerank(query, hybrid_pool, all_chunks)[0]
        rerank_hit = "HIT" if chunk_sources[reranked_top[0]] == expected_source else "miss"

        print(f"  {query:<45s} {vec_hit:>8s} {bm25_hit:>8s} {hybrid_hit:>8s} {rerank_hit:>8s}")

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print("  Advanced RAG techniques:")
    print("    1. BM25 keyword search catches exact term matches")
    print("    2. Hybrid search (vector + BM25 + RRF) combines both signals")
    print("    3. Reranking scores candidates more carefully with cross-attention")
    print("    4. HyDE bridges the query-document vocabulary gap")
    print("    5. Parent-child chunking: precise search, rich context")
    print("    6. Faithfulness evaluation catches hallucinated claims")
    print("\n  In production:")
    print("    - Replace TF-IDF with neural embeddings")
    print("    - Replace the simple reranker with a cross-encoder model")
    print("    - Replace HyDE templates with actual LLM hypothesis generation")
    print("    - Add metadata filtering before search")
    print("    - Evaluate with Recall@k and faithfulness on a test set")
