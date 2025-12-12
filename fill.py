import os
import sys
import typer

sys.path.append(os.getcwd())

from database import DatabaseWrapper
from domain.availabilities import Availabilities
from source.availabilities import get_availabilities

app = typer.Typer()

def update_availabilities(db: DatabaseWrapper, force_backfill: bool = False):
    """
    Update availabilities data from source to database.
    """
    print("\n--- Updating Availabilities ---")
    
    try:
        domain = Availabilities(db)
        print("Initialized Availabilities domain.")
    except Exception as e:
        print(f"Error initializing Availabilities domain: {e}")
        return

    print("Checking latest ingestion timestamp in database...")
    
    fetch_after = None
    should_fetch = True
    
    try:
        latest_generated = domain.latest_data_generated()
        if latest_generated:
            print(f"Latest data_generated in DB: {latest_generated}")
            fetch_after = latest_generated
        else:
            print("No existing data found (or empty table).")
            if force_backfill:
                print("Backfill forced by flag. Fetching all history.")
                fetch_after = None
            elif sys.stdin.isatty():
                if typer.confirm("Database is empty. Do you want to perform a full backfill?"):
                    print("Starting full backfill...")
                    fetch_after = None
                else:
                    print("Backfill cancelled by user.")
                    should_fetch = False
            else:
                print("Skipping backfill: Database is empty, no confirmation flag provided, and not interactive.")
                should_fetch = False
                
    except Exception as e:
        print(f"Error checking latest data_generated: {e}")

    if not should_fetch:
        print("Skipping data fetch.")
        return

    print(f"Fetching data from source (after {fetch_after})...")
    try:
        new_data = get_availabilities(after=fetch_after)
        
        count = len(new_data)
        print(f"Fetched {count} new rows.")
        
        if count > 0:
            print("Pushing to database...")
            domain.push_new_rows(new_data)
            print("Done.")
        else:
            print("No new data to push.")
            
    except Exception as e:
        print(f"Error fetching or pushing data: {e}")


@app.command()
def fill(
    db_name: str = typer.Option(..., "--db", help="The name of the D1 database to connect to."),
    force_backfill: bool = typer.Option(False, "--force-backfill", help="Force backfill without confirmation if DB is empty.")
):
    """
    Updates the database with the latest data. Creates the tables if necessary.
    """
    print(f"--- Starting Data Update for DB: {db_name} ---")
    
    try:
        db = DatabaseWrapper(database_name=db_name)
        print("Connected to database.")
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise typer.Exit(code=1)

    update_availabilities(db, force_backfill)
    
    print("\n--- Update Completed ---")

if __name__ == "__main__":
    app()
