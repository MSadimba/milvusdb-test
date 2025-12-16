# Milvus Vector Database – Responsible AI (RAI) Testbed

## Overview

This repository contains a **tested Milvus (Zilliz Cloud) vector database integration** designed to support **Responsible AI (RAI) documents** and **customer-specific contextual data**.

The goal of this work is to validate that a selected vector database provider (Milvus via Zilliz Cloud):

* can be connected to reliably,
* can store and retrieve vectorized documents,
* supports isolation of customer data,
* and works end-to-end as a backend component for future **LLM / RAG pipelines**.

All functionality is verified through **automated unit tests**, which serve as the Definition of Done for the associated Jira task.

---

## Purpose

The purpose of this project is to ensure that the chosen vector database:

* is suitable for **long-term use**,
* supports **Responsible AI knowledge bases**,
* and can act as a **production-ready backend** for LLM-powered applications.

Specifically, this repository proves that:

* RAI documents can be embedded, stored, indexed, and retrieved.
* Customer-specific data can be stored and queried in isolation.
* An LLM-equivalent workflow (embed → store → search) functions correctly.

---

## Architecture Overview

### Key Components

```text
app/
  vector_db/
    embedder.py        # Deterministic embedding stand-in (test-only)
    schemas.py         # Milvus collection schemas
    milvus_client.py   # Milvus/Zilliz DB wrapper
tests/
  test_milvus_connection.py
  test_milvus_write_retrieve.py
  test_milvus_customer_isolation.py
requirements.txt
pytest.ini
```

---

## Embedding Strategy (Important)

### Deterministic Embedding Stand-In

This project intentionally **does not use a real AI embedding model** during testing.

Instead, it uses a **deterministic embedding stand-in**, which:

* converts text into numeric vectors of fixed length,
* always produces the same vector for the same text,
* runs locally with no external dependencies.

This approach allows us to:

* reliably test vector database behavior,
* avoid flaky tests caused by external AI services,
* focus on validating database functionality rather than model quality.

In production, this embedder can be replaced with a real LLM embedding model without changing the database logic.

---

## Data Model

### RAI Documents Collection

Stores Responsible AI–related documents.

Fields:

* `id` (VARCHAR, primary key)
* `text` (document content)
* `source` (e.g. regulation, paper, policy)
* `doc_id` (external reference ID)
* `embedding` (vector representation)

### Customer Context Collection

Stores customer-specific contextual information.

Fields:

* `id` (VARCHAR, primary key)
* `customer_id` (used for isolation/filtering)
* `text` (customer-specific content)
* `embedding` (vector representation)

---

## Milvus Client

The `MilvusVectorDB` class provides a minimal, clear interface for:

* connecting to Milvus/Zilliz Cloud,
* creating collections if they do not exist,
* creating and loading vector indexes,
* inserting RAI and customer vectors,
* performing similarity search with optional filters.

This wrapper is intentionally lightweight to make future integration with RAG pipelines straightforward.

---

## Unit Tests (Definition of Done)

All functionality is validated via **pytest**.

### Tests Included

1. **Connection Test**

   * Verifies connection to Milvus/Zilliz Cloud.
   * Ensures required collections are created and indexed.

2. **RAI Write & Retrieve Test**

   * Inserts multiple RAI documents.
   * Executes vector similarity search.
   * Confirms relevant documents are retrieved.

3. **Customer Isolation Test**

   * Inserts data for multiple customers.
   * Searches using a `customer_id` filter.
   * Confirms that results are isolated per customer.

These tests directly satisfy the Jira acceptance criteria:

* an LLM can add data to the DB,
* an LLM can retrieve data from the DB,
* all unit tests pass successfully.

---

## Setup Instructions

### Prerequisites

* Python 3.10+
* Milvus/Zilliz Cloud account

### Environment Variables

Set the following before running tests:

```bash
export MILVUS_URI="https://<your-zilliz-endpoint>"
export MILVUS_TOKEN="<your-api-token>"
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Run Tests

```bash
pytest -q
```

Expected result:

```text
3 passed in XX.XXs
```

Warnings about unknown pytest markers are non-blocking.

---

## Why This Matters

This repository demonstrates that:

* Milvus (via Zilliz Cloud) is a viable long-term vector database choice.
* The backend can safely support Responsible AI knowledge bases.
* Customer data can be handled in a multi-tenant–safe way.
* The system is ready for integration with real LLM embedding models and RAG workflows.

---

## Next Steps (Optional)

* Add CI (GitHub Actions) to run tests automatically.
* Replace the deterministic embedder with a production embedding model.
* Expand schemas with additional RAI metadata.
* Integrate retrieval into a full RAG pipeline.

---

## Status

**All tests passing**
**Jira Definition of Done met**
**Ready for further integration**

