CREATE TABLE IF NOT EXISTS sessions (
    session_id UUID PRIMARY KEY,
    user_id    TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS messages (
    id            SERIAL PRIMARY KEY,
    session_id    UUID REFERENCES sessions(session_id) ON DELETE CASCADE,
    user_id       TEXT,
    role          TEXT CHECK (role IN ('user', 'assistant')),
    content       TEXT,
    message_index INTEGER,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS archived_messages (
    id             SERIAL PRIMARY KEY,
    session_id     UUID REFERENCES sessions(session_id) ON DELETE CASCADE,
    user_id        TEXT,
    role           TEXT CHECK (role IN ('user', 'assistant')),
    content        TEXT,
    message_index  INTEGER,
    compression_id INTEGER,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS compressions (
    id                  SERIAL PRIMARY KEY,
    session_id          UUID REFERENCES sessions(session_id) ON DELETE CASCADE,
    summary             TEXT,
    message_index_start INTEGER,
    message_index_end   INTEGER,
    metadata            JSONB,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_session_index  ON messages(session_id, message_index);
CREATE INDEX IF NOT EXISTS idx_archived_session_index  ON archived_messages(session_id, message_index);
CREATE INDEX IF NOT EXISTS idx_compressions_session_ts ON compressions(session_id, created_at);
