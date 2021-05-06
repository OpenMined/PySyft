-- We could also handle this the with storage plugin itself, but sense this only needs to get run once the very first time using separate sql statements
CREATE TABLE IF NOT EXISTS metadata (
    wallet_id VARCHAR(64) NOT NULL,
    value BYTEA NOT NULL,
    PRIMARY KEY(wallet_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_metadata_values ON metadata(wallet_id, value);
CREATE UNIQUE INDEX IF NOT EXISTS ux_metadata_wallet_id ON metadata(wallet_id);

CREATE TABLE IF NOT EXISTS items(
    wallet_id VARCHAR(64) NOT NULL,
    id BIGSERIAL NOT NULL,
    type BYTEA NOT NULL,
    name BYTEA NOT NULL,
    value BYTEA NOT NULL,
    key BYTEA NOT NULL,
    PRIMARY KEY(wallet_id, id)
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_items_wallet_id_id ON items(wallet_id, id);
CREATE UNIQUE INDEX IF NOT EXISTS ux_items_type_name ON items(wallet_id, type, name);

CREATE TABLE IF NOT EXISTS tags_encrypted(
    wallet_id VARCHAR(64) NOT NULL,
    name BYTEA NOT NULL,
    value BYTEA NOT NULL,
    item_id BIGINT NOT NULL,
    PRIMARY KEY(wallet_id, name, item_id),
    FOREIGN KEY(wallet_id, item_id)
        REFERENCES items(wallet_id, id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_tags_encrypted_name ON tags_encrypted(wallet_id, name);
CREATE INDEX IF NOT EXISTS ix_tags_encrypted_value ON tags_encrypted(wallet_id, sha256(value));
CREATE INDEX IF NOT EXISTS ix_tags_encrypted_wallet_id_item_id ON tags_encrypted(wallet_id, item_id);

CREATE TABLE IF NOT EXISTS tags_plaintext(
    wallet_id VARCHAR(64) NOT NULL,
    name BYTEA NOT NULL,
    value TEXT NOT NULL,
    item_id BIGINT NOT NULL,
    PRIMARY KEY(wallet_id, name, item_id),
    FOREIGN KEY(wallet_id, item_id)
        REFERENCES items(wallet_id, id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_tags_plaintext_name ON tags_plaintext(wallet_id, name);
CREATE INDEX IF NOT EXISTS ix_tags_plaintext_value ON tags_plaintext(wallet_id, value);
CREATE INDEX IF NOT EXISTS ix_tags_plaintext_wallet_id_item_id ON tags_plaintext(wallet_id, item_id);