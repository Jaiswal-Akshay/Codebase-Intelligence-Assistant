import psycopg2

def get_connection():
    return psycopg2.connect(
        database="application",
        user="admin"
    )