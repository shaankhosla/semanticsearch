import os
import time

import requests  # type: ignore
from dotenv import load_dotenv  # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from openai import OpenAI  # type: ignore
from postgres import PostgresClient
from sklearn.decomposition import PCA  # type: ignore

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=3)


# get book from gutenberg
title_url = {
    "inverted pyramid": "https://www.gutenberg.org/cache/epub/72392/pg72392.txt"
}
title_text_map = {}
for title, url in title_url.items():
    response = requests.get(url)
    book_text = response.text

    title_text_map[title] = book_text


# use langchain to do document chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=40,
    length_function=len,
    is_separator_regex=False,
)
documents = []
for title, text in title_text_map.items():
    text_chunks = text_splitter.split_text(text)[:1000]

    embeddings = openai_client.embeddings.create(
        input=text_chunks,
        model="text-embedding-ada-002",
    ).data
    embeddings = [e.embedding for e in embeddings]

    pca = PCA(n_components=50)
    reduced_embeddings = pca.fit_transform(embeddings)

    for i, text_chunk in enumerate(text_chunks):
        documents.append(
            {
                "title": title,
                "text": text_chunk,
                "large_embedding": embeddings[i],
                "small_embedding": reduced_embeddings[i],
            }
        )


# compare sizes
postgres_client = PostgresClient(large_embedding_size=1536, small_embedding_size=50)
postgres_client.delete_tables()
postgres_client.create_tables()
postgres_client.add_data(data=documents)
size = postgres_client.get_vector_column_size()
for column, size_bytes in size.items():
    print(f"Column {column} has size {size_bytes/1048576} MB")
print("\n")

# search over table

quote = "Where was Rod Norquay sitting?"
quote_embedding = (
    openai_client.embeddings.create(
        input=[quote],
        model="text-embedding-ada-002",
    )
    .data[0]
    .embedding
)

st_time = time.time()
results = postgres_client.search_db(
    query_vec=quote_embedding,
    column="large_embedding",
)
print(f"Result using large embeddings took {time.time() -st_time:.2f}:\n")
print(results[0]["text"])
print("\n" * 5)

reduced_quote_embedding = pca.transform([quote_embedding])[0]
st_time = time.time()
results = postgres_client.search_db(
    query_vec=reduced_quote_embedding,
    column="small_embedding",
)
print(f"Result using small embeddings took {time.time() -st_time:.2f}:\n")
print(results[0]["text"])
print("\n" * 5)
