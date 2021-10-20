"""Implements a simple sqlite3 database with helper functions for adding rows

By Daryl Lim
"""

import os
import sys
import pandas
import sqlite3 as sql3


class SQL3DB(object):
    def __init__(self, db_path, schema):
        self.schema = schema
        self.db_path = db_path
        dir_name, _ = (os.path.dirname(db_path), os.path.basename(db_path))
        if not os.path.exists(dir_name):
            print("Creating %s" % dir_name)
            os.makedirs(dir_name)

        self.primary_key = schema["primary_key"]

        if not os.path.exists(db_path):
            print("%s does not exist, creating new database" % db_path)
            conn = sql3.connect("%s" % db_path)

            c = conn.cursor()
            sql = self.build_initial_sql_query_from_schema(self.schema)
            c.execute(sql)
            conn.commit()
            conn.close()

        entries_df = self.read_df_from_db(self.db_path)
        print(
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
                print("Error in SQL transaction")
                print(sys.exc_info())
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
