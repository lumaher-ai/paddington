import asyncio

import asyncpg

DATABASE_URL = "postgresql://paddington:paddington_dev_password@localhost:5432/paddington"


async def main() -> None:
    # Conectarse a la DB
    conn = await asyncpg.connect(DATABASE_URL)

    try:
        # Crear una tabla simple
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS test_users (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        # Insertar datos
        await conn.execute(
            "INSERT INTO test_users (name) VALUES ($1)",
            "Luisa",
        )

        # Leer datos
        rows = await conn.fetch("SELECT id, name, created_at FROM test_users")

        print(f"Found {len(rows)} users:")
        for row in rows:
            print(f"  {row['id']}: {row['name']} (created at {row['created_at']})")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
