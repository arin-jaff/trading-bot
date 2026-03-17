"""Database models for the Trump Mentions Trading Bot."""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text,
    ForeignKey, JSON, Table, UniqueConstraint, Index
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

# Association table for market terms (many-to-many)
market_term_association = Table(
    'market_term_association',
    Base.metadata,
    Column('market_id', Integer, ForeignKey('markets.id'), primary_key=True),
    Column('term_id', Integer, ForeignKey('terms.id'), primary_key=True),
)


class Market(Base):
    """Kalshi Trump Mentions market."""
    __tablename__ = 'markets'

    id = Column(Integer, primary_key=True)
    kalshi_ticker = Column(String(100), unique=True, nullable=False)
    kalshi_event_ticker = Column(String(100), nullable=False)
    title = Column(String(500), nullable=False)
    subtitle = Column(String(500))
    market_type = Column(String(50))  # e.g., 'trump_mentions'
    status = Column(String(50))  # active, closed, settled
    yes_price = Column(Float)
    no_price = Column(Float)
    volume = Column(Integer)
    open_interest = Column(Integer)
    close_time = Column(DateTime)
    expiration_time = Column(DateTime)
    result = Column(String(10))  # yes, no, null
    raw_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    terms = relationship('Term', secondary=market_term_association, back_populates='markets')
    price_history = relationship('PriceSnapshot', back_populates='market', cascade='all, delete-orphan')
    trades = relationship('Trade', back_populates='market', cascade='all, delete-orphan')

    __table_args__ = (
        Index('idx_market_status', 'status'),
        Index('idx_market_event', 'kalshi_event_ticker'),
    )


class Term(Base):
    """A word or phrase tracked across Trump Mentions markets.

    Supports single words ('tariff'), multi-word phrases ('who are you with'),
    and compound terms ('who are you with / where are you from').
    """
    __tablename__ = 'terms'

    id = Column(Integer, primary_key=True)
    term = Column(String(500), unique=True, nullable=False)
    normalized_term = Column(String(500), nullable=False)  # lowercase, stripped
    is_compound = Column(Boolean, default=False)  # True for "X / Y" style terms
    sub_terms = Column(JSON)  # List of individual terms if compound
    category = Column(String(100))  # optional grouping
    total_occurrences = Column(Integer, default=0)
    trend_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    markets = relationship('Market', secondary=market_term_association, back_populates='terms')
    occurrences = relationship('TermOccurrence', back_populates='term', cascade='all, delete-orphan')
    predictions = relationship('TermPrediction', back_populates='term', cascade='all, delete-orphan')

    __table_args__ = (
        Index('idx_term_normalized', 'normalized_term'),
    )


class Speech(Base):
    """A Trump speech, rally, press conference, or public appearance."""
    __tablename__ = 'speeches'

    id = Column(Integer, primary_key=True)
    source = Column(String(100), nullable=False)  # youtube, whitehouse, cspan, etc.
    source_url = Column(String(1000))
    source_id = Column(String(200))  # external ID from source
    title = Column(String(1000), nullable=False)
    speech_type = Column(String(100))  # rally, press_conference, interview, etc.
    date = Column(DateTime, nullable=False)
    duration_seconds = Column(Integer)
    transcript = Column(Text)
    transcript_source = Column(String(100))  # how we got the transcript
    word_count = Column(Integer)
    is_processed = Column(Boolean, default=False)
    raw_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    occurrences = relationship('TermOccurrence', back_populates='speech', cascade='all, delete-orphan')

    __table_args__ = (
        UniqueConstraint('source', 'source_id', name='uq_speech_source'),
        Index('idx_speech_date', 'date'),
    )


class TermOccurrence(Base):
    """Tracks when a term appeared in a specific speech."""
    __tablename__ = 'term_occurrences'

    id = Column(Integer, primary_key=True)
    term_id = Column(Integer, ForeignKey('terms.id'), nullable=False)
    speech_id = Column(Integer, ForeignKey('speeches.id'), nullable=False)
    count = Column(Integer, default=0)
    context_snippets = Column(JSON)  # list of surrounding text excerpts
    created_at = Column(DateTime, default=datetime.utcnow)

    term = relationship('Term', back_populates='occurrences')
    speech = relationship('Speech', back_populates='occurrences')

    __table_args__ = (
        UniqueConstraint('term_id', 'speech_id', name='uq_term_speech'),
        Index('idx_occurrence_term', 'term_id'),
        Index('idx_occurrence_speech', 'speech_id'),
    )


class TrumpEvent(Base):
    """Upcoming Trump public appearances / events."""
    __tablename__ = 'trump_events'

    id = Column(Integer, primary_key=True)
    title = Column(String(500), nullable=False)
    event_type = Column(String(100))  # rally, press_conference, state_dinner, etc.
    location = Column(String(500))
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    is_live = Column(Boolean, default=False)
    is_confirmed = Column(Boolean, default=True)
    source_url = Column(String(1000))
    notes = Column(Text)
    topics = Column(JSON)  # expected topics based on context
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_event_start', 'start_time'),
    )


class TermPrediction(Base):
    """ML model prediction for term usage likelihood."""
    __tablename__ = 'term_predictions'

    id = Column(Integer, primary_key=True)
    term_id = Column(Integer, ForeignKey('terms.id'), nullable=False)
    event_id = Column(Integer, ForeignKey('trump_events.id'), nullable=True)
    model_name = Column(String(100), nullable=False)
    probability = Column(Float, nullable=False)
    confidence = Column(Float)
    reasoning = Column(Text)
    features_used = Column(JSON)
    prediction_date = Column(DateTime, default=datetime.utcnow)
    target_date = Column(DateTime)
    was_correct = Column(Boolean)  # filled in after settlement
    created_at = Column(DateTime, default=datetime.utcnow)

    term = relationship('Term', back_populates='predictions')

    __table_args__ = (
        Index('idx_prediction_term', 'term_id'),
        Index('idx_prediction_date', 'prediction_date'),
    )


class PriceSnapshot(Base):
    """Historical price data for markets."""
    __tablename__ = 'price_snapshots'

    id = Column(Integer, primary_key=True)
    market_id = Column(Integer, ForeignKey('markets.id'), nullable=False)
    yes_price = Column(Float)
    no_price = Column(Float)
    volume = Column(Integer)
    open_interest = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

    market = relationship('Market', back_populates='price_history')

    __table_args__ = (
        Index('idx_snapshot_market_time', 'market_id', 'timestamp'),
    )


class Trade(Base):
    """Trades placed on Kalshi."""
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    market_id = Column(Integer, ForeignKey('markets.id'), nullable=False)
    kalshi_order_id = Column(String(100))
    side = Column(String(10), nullable=False)  # yes, no
    action = Column(String(10), nullable=False)  # buy, sell
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    fill_price = Column(Float)
    status = Column(String(50))  # pending, filled, cancelled
    pnl = Column(Float)
    strategy = Column(String(100))  # which strategy triggered this
    reasoning = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    market = relationship('Market', back_populates='trades')

    __table_args__ = (
        Index('idx_trade_market', 'market_id'),
        Index('idx_trade_status', 'status'),
    )


class ModelVersion(Base):
    """Tracks each iteration of the TrumpGPT model."""
    __tablename__ = 'model_versions'

    id = Column(Integer, primary_key=True)
    version = Column(String(20), unique=True, nullable=False)  # "1.0.0", "1.0.1"
    model_type = Column(String(50), nullable=False)  # "markov_chain", "colab_llm"
    markov_order = Column(Integer)
    corpus_size = Column(Integer)  # number of speeches used
    corpus_word_count = Column(Integer)  # total words in training corpus
    training_duration_seconds = Column(Float)
    simulation_count = Column(Integer)  # Monte Carlo runs
    prediction_count = Column(Integer)  # terms predicted
    metrics = Column(JSON)  # {avg_probability, unique_terms, etc.}
    artifact_path = Column(String(500))  # path to saved model file
    is_active = Column(Boolean, default=True)  # currently used model
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_model_version', 'version'),
    )


class BotConfig(Base):
    """Bot configuration and state."""
    __tablename__ = 'bot_config'

    id = Column(Integer, primary_key=True)
    key = Column(String(200), unique=True, nullable=False)
    value = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
