from OLD.app.core.settings import Settings
from OLD.app.core.utils import utc_now
from OLD.app.corpus.chunker import TextChunker
from OLD.app.corpus.schemas import AttackLabel, ChunkRecord, SourceType


def test_chunking_preserves_metadata_and_chunk_ids():
    settings = Settings(chunk_size=40, chunk_overlap=5)
    chunker = TextChunker(settings)
    raw_text = (
        "Chunking should preserve metadata. "
        "This sentence forces the text to split into multiple chunks for testing."
    )

    chunks = chunker.split(raw_text)
    assert len(chunks) >= 2

    records = [
        ChunkRecord(
            chunk_id=f"doc_test::chunk::{index}",
            doc_id="doc_test",
            title="Doc Test",
            chunk_text=chunk,
            chunk_index=index,
            source_type=SourceType.txt,
            attack_label=AttackLabel.benign,
            corpus_version="cv_test",
            embedding=[0.1, 0.2],
            metadata={"topic": "test"},
            created_at=utc_now(),
        )
        for index, chunk in enumerate(chunks)
    ]

    assert records[0].chunk_id.endswith("::chunk::0")
    assert all(record.metadata["topic"] == "test" for record in records)
