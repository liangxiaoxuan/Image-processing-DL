import pymysql
import pymysql.cursors
import uuid


"""
Add the model to DB
"""

host = 'localhost'
password = '1234'
db = 'project'
user = 'root'

def register_model(des, repo, sha1, author, is_attack, checkpoint):
    """
    :param id: each model added
    :param des: description of the model
    :param repo: URL address
    :param sha1: version of commit
    :param author: Name of the author
    :param is_attack: 1: attacked; 0: non-attack
    :param checkpoint: checkpoint address
    :return: model id

    """

    conn = pymysql.connect(host=host,
                           user=user,
                           password=password,
                           db=db)
    cursor = conn.cursor()

    sql = 'INSERT INTO model(description, repo, sha1, author, is_attack, checkpoint) VALUES (%s,%s,%s,%s,%s,%s)'
    data = (des, repo, sha1, author, is_attack, checkpoint)
    cursor.execute(sql, data)
    id = cursor.lastrowid

    conn.commit()
    cursor.close()
    conn.close()

    return id
    
    
 def get_model_checkpoint(checkpoint):

    conn = pymysql.connect(host=host,
                           user=user,
                           password=password,
                           db=db)
    cursor = conn.cursor()

    sql = "SELECT ID FROM model WHERE model.checkpoint = '%s'" % checkpoint
    cursor.execute(sql)
    result = cursor.fetchone()
    conn.commit()
    cursor.close()
    conn.close()

    return result[0]

def test():
    print("abc")

    
    
 if __name__ == '__main__':
 
