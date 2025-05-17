import enum
from sqlalchemy import Column as SAColumn, Table, Text, DateTime, Date, Boolean, Numeric, CHAR, ForeignKey, UniqueConstraint, text, Integer
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID, ENUM as PGEnum
from database import Base

# Define the answer_format enum type
class AnswerFormat(enum.Enum):
    text = 'text'
    date = 'date'
    boolean = 'boolean'
    currency = 'currency'

answer_format_enum = PGEnum(
    AnswerFormat,
    name='answer_format',
    create_type=True
)

# Documents table
class Document(Base):
    __tablename__ = 'documents'

    id = SAColumn(UUID(as_uuid=True), primary_key=True, server_default=text('gen_random_uuid()'))
    file_ref = SAColumn(Text, nullable=False)
    filename = SAColumn(Text)
    uploaded_at = SAColumn(DateTime(timezone=True), server_default=func.now())

# Rows table
class Row(Base):
    __tablename__ = 'rows'

    id = SAColumn(UUID(as_uuid=True), primary_key=True, server_default=text('gen_random_uuid()'))
    name = SAColumn(Text)
    created_at = SAColumn(DateTime(timezone=True), server_default=func.now())

# Association table for rows ↔ documents
row_documents = Table(
    'row_documents',
    Base.metadata,
    SAColumn('row_id', UUID(as_uuid=True), ForeignKey('rows.id', ondelete='CASCADE'), primary_key=True),
    SAColumn('document_id', UUID(as_uuid=True), ForeignKey('documents.id', ondelete='CASCADE'), primary_key=True),
)

# Columns table
class Column(Base):
    __tablename__ = 'columns'

    id = SAColumn(UUID(as_uuid=True), primary_key=True, server_default=text('gen_random_uuid()'))
    label = SAColumn(Text, nullable=False)
    prompt = SAColumn(Text, nullable=False)
    format = SAColumn(answer_format_enum, nullable=False)

# Cells table
class Cell(Base):
    __tablename__ = 'cells'

    id = SAColumn(UUID(as_uuid=True), primary_key=True, server_default=text('gen_random_uuid()'))
    row_id = SAColumn(UUID(as_uuid=True), ForeignKey('rows.id', ondelete='CASCADE'), nullable=False)
    column_id = SAColumn(UUID(as_uuid=True), ForeignKey('columns.id', ondelete='CASCADE'), nullable=False)
    answer_text = SAColumn(Text)
    answer_date = SAColumn(Date)
    answer_boolean = SAColumn(Boolean)
    answer_amount = SAColumn(Numeric(18, 4))
    answer_currency = SAColumn(CHAR(3))
    computed_at = SAColumn(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('row_id', 'column_id', name='uix_row_column'),
    )

class Chunk(Base):
    __tablename__ = 'chunks'

    id = SAColumn(UUID(as_uuid=True), primary_key=True, server_default=text('gen_random_uuid()'))

    row_id      = SAColumn(UUID(as_uuid=True), ForeignKey('rows.id', ondelete='CASCADE'), nullable=False)
    document_id = SAColumn(UUID(as_uuid=True), ForeignKey('documents.id', ondelete='CASCADE'), nullable=False)

    chunk_index = SAColumn(Integer, nullable=False)

    text        = SAColumn(Text, nullable=False)

    created_at  = SAColumn(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('row_id', 'document_id', 'chunk_index', name='uix_chunk_identity'),
    )