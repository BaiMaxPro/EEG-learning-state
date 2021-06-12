import os
import shutil
import datetime
import sys

import pymysql
import pandas as pd

from muselsl import stream,list_muses
from Process import ScoreF

# muses = list_muses()

if len(sys.argv) < 3:
		print ('arg1: id\narg2: classname')
		sys.exit(-1)
id = sys.argv[1]
classname = sys.argv[2]

conn = pymysql.connect(
            host='frp.smartbai.cn',  
            port=3333,
            user='root',  
            passwd='calvin',
            db='user',  
            charset='utf8',  
            cursorclass=pymysql.cursors.DictCursor
        )

cur = conn.cursor()

conn.commit()

try:
    stream('00:55:da:b7:5d:a6')
except AttributeError:
    print('Maybe it is streaming!')



while True:
    date = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    os.system('muselsl record --duration 30 -f ./record.csv')
    date = str(date)
    score = ScoreF('./record.csv')
    # time = date.strftime("%m-%d--%H:%M:%S")
    # shutil.move('./record.csv', f'.\\used_data\\Record-{time}-{int(score)}')
    # os.system(f'mv ./record.csv ./used_data/record-{time}-{int(score)}')

    data = {
        'iduser':str(id),
        'class':str(classname),
        'date':date,
        'score':float(score)
    }
    # print(data)
    # sql_write = pd.DataFrame(data,index=False)
    # print(sql_write)
    # sql_write.to_sql('user_history',conn,index=False)
    sql = f'''INSERT INTO user_history (iduser,class,date,score) 
    VALUES({data['iduser']},{data['class']},'{data['date']}',{data['score']})'''
    cur.execute(sql)
    conn.commit()

    os.remove('./record.csv')




