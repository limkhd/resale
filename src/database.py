"""Implements a simple sqlite3 database with helper functions for adding rows

By Daryl Lim
"""

import logging
import os
import sys
import pandas
import sqlite3 as sql3

logger = logging.getLogger(__name__)


class SQL3DB(object):
    def __init__(self, db_path, schema):
        self.schema = schema
        self.db_path = db_path
        dir_name, _ = (os.path.dirname(db_path), os.path.basename(db_path))
        if not os.path.exists(dir_name):
            logger.info("Creating %s" % dir_name)
            os.makedirs(dir_name)

        self.primary_key = schema["primary_key"]

        table_in_db = True

        if not os.path.exists(db_path):
            # DB file doesn't exist
            logger.info("%s does not exist, creating new database" % db_path)
            table_in_db = False

        else:
            # DB file exists but table was not created
            with sql3.connect("%s" % db_path) as conn:
                c = conn.cursor()
                sql = "SELECT name FROM sqlite_master WHERE type='table' AND name='database'"

                try:
                    res = c.execute(sql)
                    if res.fetchone() is None:
                        table_in_db = False
                except:
                    logger.error("Error in SQL transaction")
                    logger.error(sys.exc_info())
                    sys.exit(1)

        if not table_in_db:
            logger.info("Creating 'database' table in sqlite DB")
            with sql3.connect("%s" % db_path) as conn:
                c = conn.cursor()
                sql = self.build_initial_sql_query_from_schema(self.schema)
                c.execute(sql)

        # Get all entries in pandas DataFrame
        entries_df = self.read_df_from_db(self.db_path)
        logger.info(
            "%d entries currently in database table in %s" % (len(entries_df), db_path)
        )

    def build_initial_sql_query_from_schema(self, schema):
        """Generates SQL code to create a table with specified nested schema"""

        sql = "CREATE TABLE database ("
        for c in schema["columns"]:
            sql += '"%s" %s, ' % (c, schema["columns"][c])
        sql += "PRIMARY KEY(%s)" % ", ".join(
            ['"%s"' % x for x in schema["primary_key"]]
        )
        sql += ")"
        return sql

    def key_in_db(self, primary_key_values):
        is_in_db = None
        with sql3.connect("%s" % self.db_path) as conn:
            c = conn.cursor()
            sql = ("SELECT COUNT(1) FROM database WHERE %s") % (
                " AND ".join(['"%s" = ?' % x for x in self.primary_key])
            )

            try:
                res = c.execute(sql, primary_key_values)
                is_in_db = res.fetchone()[0] == 1
            except:
                logger.error("Error in SQL transaction")
                logger.error(sys.exc_info())
                sys.exit(1)

        return is_in_db

    def store_dict_in_db(self, record_dict):
        """Store a record_dict into DB where record_dict keys are column names."""

        cols, values = zip(*record_dict.items())
        sql = "INSERT INTO database (%s) VALUES (%s)" % (
            ",".join(cols),
            ",".join(["?" for _ in range(len(values))]),
        )

        try:
            with sql3.connect("%s" % self.db_path) as conn:
                c = conn.cursor()
                c.execute(sql, values)
                conn.commit()
        except:
            print("Error writing to SQL database")
            print(sys.exc_info())
            sys.exit(1)

    def read_df_from_db(self, path):
        """Loads table into pandas DataFrame and returns it"""
        try:
            conn = sql3.connect(path)
            table = pandas.read_sql_query("SELECT * from database", conn)
            conn.close()
        except:
            print("Error in SQL transaction")
            print(sys.exc_info())
            sys.exit(1)

        return table
