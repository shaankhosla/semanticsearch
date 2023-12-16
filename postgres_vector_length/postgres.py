import os
from typing import Literal

import numpy as np
import psycopg  # type: ignore
from dotenv import load_dotenv  # type: ignore
from pgvector.psycopg import register_vector  # type: ignore
from psycopg.rows import dict_row  # type: ignore

load_dotenv()


class PostgresClient:
    def __init__(
        self,
        large_embedding_size: int,
        small_embedding_size: int,
        postgres_database: str = "semantic_search",
    ):
        NEON_USERNAME = os.getenv("NEON_USERNAME")
        NEON_PASSWORD = os.getenv("NEON_PASSWORD")
        self.large_embedding_size = large_embedding_size
        self.small_embedding_size = small_embedding_size

        self.postgres_url = f"postgresql://{NEON_USERNAME}:{NEON_PASSWORD}@ep-still-hat-20912390.us-east-2.aws.neon.tech/{postgres_database}?sslmode=require"

    def create_postgres_connection(self):
        if hasattr(self, "postgres_client"):
            self.postgres_client.close()
        self.postgres_client = psycopg.connect(
            conninfo=self.postgres_url,
            row_factory=dict_row,
        )
        register_vector(self.postgres_client)

    def init_postgres_client(self):
        if (
            not hasattr(self, "postgres_client")
            or self.postgres_client.connection.closed
            or self.postgres_client.connection.broken
        ):
            self.create_postgres_connection()
        try:
            with self.postgres_client.cursor() as cursor:
                cursor.execute("SELECT 1")
        except (psycopg.DatabaseError, psycopg.OperationalError):
            self.create_postgres_connection()

    def create_tables(self):
        self.init_postgres_client()
        create_table_sql = f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT FROM pg_tables
                WHERE schemaname = 'public'
                AND tablename = 'search_data'
            ) THEN
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE TABLE search_data (
                    book_title VARCHAR(255) NOT NULL,
                    text TEXT NOT NULL,
                    large_embedding Vector({self.large_embedding_size}),
                    small_embedding Vector({self.small_embedding_size})
                );
                CREATE INDEX ON search_data USING hnsw (large_embedding vector_ip_ops);
                CREATE INDEX ON search_data USING hnsw (small_embedding vector_ip_ops);
            END IF;
        END
        $$;
        """

        with self.postgres_client.cursor() as cursor:
            cursor.execute(create_table_sql)
            self.postgres_client.commit()
        register_vector(self.postgres_client)

    def delete_tables(self):
        self.init_postgres_client()
        sql = """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT FROM pg_tables
                WHERE schemaname = 'public'
                AND tablename = 'search_data'
            ) THEN
                DROP TABLE search_data;
            END IF;
        END
        $$;
        """
        with self.postgres_client.cursor() as cursor:
            cursor.execute(sql)
            self.postgres_client.commit()

    def add_data(self, data: list[dict]):
        insert_query = """
            INSERT INTO search_data (book_title, text, large_embedding, small_embedding)
            VALUES (%s, %s, %s, %s)
            """
        insert_data = [
            (
                d["title"],
                d["text"],
                d["large_embedding"],
                np.array(d["small_embedding"]),
            )
            for d in data
        ]
        with self.postgres_client.cursor() as cursor:
            cursor.executemany(insert_query, insert_data)
            self.postgres_client.commit()

    def search_db(
        self,
        query_vec: list[float],
        column: Literal["large_embedding", "small_embedding"],
    ) -> list:
        self.init_postgres_client()
        query = f"""
        SELECT book_title, text
        FROM search_data
        ORDER BY {column} <-> %s
        LIMIT 1;
        """

        with self.postgres_client.cursor() as cursor:
            results = cursor.execute(
                query,
                (np.array(query_vec),),
            ).fetchall()

        return results

    def get_vector_column_size(self) -> dict:
        sql = """
        SELECT SUM(pg_column_size(large_embedding)) as large_embedding_size,
               SUM(pg_column_size(small_embedding)) as small_embedding_size
        FROM search_data;
        """
        with self.postgres_client.cursor() as cursor:
            results = cursor.execute(sql).fetchone()

        return results
