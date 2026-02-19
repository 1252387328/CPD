import json
import os
import sqlite3


def cal_path1(Task,init_strength,min_strength,init_node,path_number,des_var):
    # print(Task,init_strength,min_strength,init_node,path_number)
    try:
        with open(f'./paths/t{Task}i{init_strength}m{min_strength}/init{init_node}.json', 'r') as f:
            paths = json.load(f)[:path_number]
    except:
        print(f"t{Task}i{init_strength}m{min_strength}不存在的路径")
        directory_path = f"./paths/t{Task}i{init_strength}m{min_strength}"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            # print("目录不存在已创建")
        paths = des_var.find_paths(path_number, init_node, init_strength)
        with open(f'{directory_path}/init{init_node}.json', 'w') as file:
            json.dump(paths, file)
            print(f"路径已保存至{directory_path}/init{init_node}.json")
    return(paths)

def cal_path(Task, init_strength, min_strength, init_node, path_number, des_var):
    # Connect to the SQLite database (it will create the database file if it does not exist)
    conn = sqlite3.connect('./db/cal_path.db')
    c = conn.cursor()

    # Create table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS paths
                 (task TEXT, init_strength TEXT, min_strength TEXT, 
                  init_node INTEGER, path TEXT)''')

    # Attempt to retrieve existing paths
    c.execute('''SELECT path FROM paths 
                 WHERE task = ? AND init_strength = ? AND min_strength = ? AND init_node = ?''',
              (Task, init_strength, min_strength, init_node))
    load_paths_json = c.fetchall()

    if load_paths_json:
        # Convert retrieved paths from tuples to list
        paths = json.loads(load_paths_json[0][0])
    else:
        # If paths do not exist, generate and insert them
        paths = des_var.find_paths(path_number, init_node, init_strength)
        # paths = cal_path(Task,init_strength,min_strength,init_node,path_number,des_var)
        paths_json= json.dumps(paths)  # Convert list to string for storage in the database

        # Insert generated paths into the database
        c.execute('''INSERT INTO paths (task, init_strength, min_strength, init_node, path) 
                         VALUES (?, ?, ?, ?, ?)''',
                      (
                      Task, init_strength, min_strength, init_node, paths_json))  # Storing path as a string

        conn.commit()
        print(f"路径已保存至数据库")

    # Close the connection
    conn.close()
    return paths