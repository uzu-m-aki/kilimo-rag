import os
import sys
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# PostgreSQL connection string
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://raguser:ragpassword@localhost:5432/ragdb")

def create_tables():
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
            conn.commit()
            conn.execute(text('DROP TABLE IF EXISTS chat_messages'))
            conn.commit()
            create_table_sql = """
            CREATE TABLE chat_messages (
                message_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                session_id UUID NOT NULL,
                user_message TEXT,
                system_message TEXT,
                role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX idx_chat_messages_session_id ON chat_messages(session_id);
            CREATE INDEX idx_chat_messages_timestamp ON chat_messages(timestamp);
            CREATE INDEX idx_chat_messages_role ON chat_messages(role);
            """
            conn.execute(text(create_table_sql))
            conn.commit()
            logger.info("Tables created successfully")
        engine.dispose()
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    logger.info("Initializing database...")
    create_tables()
