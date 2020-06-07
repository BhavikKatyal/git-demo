"""Main module for App Engine app."""

# Import Library Dependencies
from flask import Flask, jsonify, request, jsonify
import json

#Import Local Library Dependencies
#from apiserver.modelserve import Recommendations
from google.cloud import storage
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

# Initialise Flask Application
app = Flask(__name__)

# Global default variables
DEFAULT_RECS = 5
GOOGLE_APPLICATION_CREDENTIALS="maverick-ai-rec-engine-dev-a790f32a7a9b.json"

# # Start jobs which need to be run before the first request
#@app.before_first_request
def User_item_score1(user,nrecs_int):
    courses=pd.read_pickle('courses.pkl')
    enr=pd.read_pickle('enrollments.pkl')
    users=pd.read_pickle('users.pkl') 
    enr_avg=pd.read_pickle('enr_avg.pkl')
    Mean=pd.read_pickle('Mean.pkl')
    final_user=pd.read_pickle('final_user.pkl')
    final_courses=pd.read_pickle('final_courses.pkl')
    similarity_with_user=pd.read_pickle('similarity_with_user.pkl')
    similarity_with_courses=pd.read_pickle('similarity_with_courses.pkl')
    sim_user_n_u=pd.read_pickle('sim_user_n_u.pkl')
    sim_user_n_m=pd.read_pickle('sim_user_n_m.pkl')
    Courses_user=pd.read_pickle('Courses_user.pkl')

    Courses_seen_by_user=[]
    for x in range(0,len(enr)):
        if(enr['userId'][x]==user) and (enr['Enrollment'][x]==1):
            Courses_seen_by_user.append(enr['Course_Id'][x])

    print(Courses_seen_by_user)
    a = sim_user_n_m[sim_user_n_m.index==user].values
    print(a)
    b = a.squeeze().tolist()
    print(b)
    d = Courses_user[Courses_user.index.isin(b)]
    print(d)
    l = ','.join(d.values)
    print(l)
    Courses_seen_by_similar_users = l.split(',')
    Courses_under_consideration=list(set(Courses_seen_by_similar_users)-set(list(map(str, Courses_seen_by_user))))
    print(Courses_under_consideration)
    Courses_under_consideration = list(map(int, Courses_under_consideration))
    print(Courses_under_consideration)
    score = []
    for item in Courses_under_consideration:
        c = final_courses.loc[:,item]
        d = c[c.index.isin(b)]
        f = d[d.notnull()]
        avg_user = Mean.loc[Mean['userId'] == user,'Enrollment'].values[0]
        index = f.index.values.squeeze().tolist()
        corr = similarity_with_courses.loc[user,index]
        fin = pd.concat([f, corr], axis=1)
        fin.columns = ['adg_score','correlation']
        fin['score']=fin.apply(lambda x:x['adg_score'] * x['correlation'],axis=1)
        nume = fin['score'].sum()
        print(nume)
        deno = fin['correlation'].sum()
        final_score = avg_user + (nume/deno)
        score.append(final_score)
        print(score)
    data = pd.DataFrame({'Course_Id':Courses_under_consideration,'score':score})
    print(data)
    top_5_recommendation = data.sort_values(by='score',ascending=False)
    Course_Name = top_5_recommendation.merge(courses, how='inner', on='Course_Id')
    Course_Names = Course_Name.Title.values.tolist()
    return Course_Names

def find_n_neighbours(df,n):
    order = np.argsort(df.values, axis=1)[:, :n]
    df = df.apply(lambda x: pd.Series(x.sort_values(ascending=False)
           .iloc[:n].index, 
          index=['top{}'.format(i) for i in range(1, n+1)]), axis=1)
    return df


MODEL_BUCKET='mtx-recommendation-engine'
@app.route('/data',methods=['POST'])
def matrix_generator():
      
    client = storage.Client()
    bucket = client.get_bucket(MODEL_BUCKET)
    print("Hi"+str(bucket))
    courses='courses.csv'
    users='users.csv'
    enr='Enrollments.csv'
    courses_blob = bucket.get_blob(courses)
    users_blob = bucket.get_blob(users)
    enr_blob = bucket.get_blob(enr)
    
    with open("courses.csv", "wb") as file_obj:
        courses_blob.download_to_file(file_obj)
    with open("users.csv", "wb") as file_obj:
        users_blob.download_to_file(file_obj)
    with open("enrollments.csv", "wb") as file_obj:
        enr_blob.download_to_file(file_obj)
    courses = pd.read_csv("courses.csv",encoding="Latin1")
    enr = pd.read_csv("enrollments.csv")
    users=pd.read_csv("users.csv")
    #1
    courses.to_pickle("courses.pkl")
    courses_blob = bucket.blob('courses.pkl')
    courses_blob.upload_from_filename('courses.pkl')
    #2
    enr.to_pickle("enrollments.pkl")
    enr_blob = bucket.blob('enrollments.pkl')
    enr_blob.upload_from_filename('courses.pkl')
    #3
    users.to_pickle("users.pkl")
    users_blob = bucket.blob('users.pkl')
    users_blob.upload_from_filename('users.pkl')
    Mean = enr.groupby(by="userId",as_index=False)['Enrollment'].mean()
    ##New
    Mean.to_pickle("Mean.pkl")
    Mean_blob = bucket.blob('Mean.pkl')
    Mean_blob.upload_from_filename('Mean.pkl')
    enr_avg = pd.merge(enr,Mean,on='userId')
    enr_avg['adg_rating']=enr_avg['Enrollment_x']-enr_avg['Enrollment_y']
    #4
    enr_avg.to_pickle('enr_avg.pkl')
    enr_avg_blob = bucket.blob('enr_avg.pkl')
    enr_avg_blob.upload_from_filename('enr_avg.pkl')
    
    print(users.shape[0])
    l=users.shape[0]
    final=pd.pivot_table(enr_avg,values='adg_rating',index='userId',columns='Course_Id')
    final_courses = final.fillna(final.mean(axis=0))

# Replacing NaN by user Average
    final_user = final.apply(lambda row: row.fillna(row.mean()), axis=1)
    #5
    final_user.to_pickle('final_user.pkl')
    final_user_blob = bucket.blob('final_user.pkl')
    final_user_blob.upload_from_filename('final_user.pkl')
    
    # Replacing NaN by Courses Average
    #6
    final_courses = final.fillna(final.mean(axis=0))
    final_courses.to_pickle('final_courses.pkl')
    final_courses_blob = bucket.blob('final_courses.pkl')
    final_courses_blob.upload_from_filename('final_courses.pkl')
    
    b = cosine_similarity(final_user)
    np.fill_diagonal(b, 0 )
    similarity_with_user = pd.DataFrame(b,index=final_user.index)
    similarity_with_user.columns=final_user.index
    #7
    similarity_with_user.to_pickle('similarity_with_user.pkl')
    similarity_with_user_blob = bucket.blob('similarity_with_user.pkl')
    similarity_with_user_blob.upload_from_filename('similarity_with_user.pkl')

    cosine = cosine_similarity(final_courses)
    np.fill_diagonal(cosine, 0 )
    similarity_with_courses =pd.DataFrame(cosine,index=final_courses.index)
    similarity_with_courses.columns=final_courses.index
    #8
    similarity_with_courses.to_pickle("similarity_with_courses.pkl")
    similarity_with_courses_blob = bucket.blob('similarity_with_courses.pkl')
    similarity_with_courses_blob.upload_from_filename('similarity_with_courses.pkl')
    
    
    print("Here")
    sim_user_n_u = find_n_neighbours(similarity_with_user,l)
    #9
    sim_user_n_u.to_pickle('sim_user_n_u.pkl')
    sim_user_n_u_blob = bucket.blob('sim_user_n_u.pkl')
    sim_user_n_u_blob.upload_from_filename('sim_user_n_u.pkl')
    
    
    sim_user_n_m = find_n_neighbours(similarity_with_courses,l)
    #10
    sim_user_n_m.to_pickle('sim_user_n_m.pkl')
    sim_user_n_m_blob = bucket.blob('sim_user_n_m.pkl')
    sim_user_n_m_blob.upload_from_filename('sim_user_n_m.pkl')
    
    enr_avg = enr.astype({"Course_Id": str})
    Courses_user = enr_avg.groupby(by = 'userId')['Course_Id'].apply(lambda x:','.join(x))
    Courses_user.to_pickle('Courses_user.pkl')
    Courses_user_blob = bucket.blob('Courses_user.pkl')
    Courses_user_blob.upload_from_filename('Courses_user.pkl')
    # Make an authenticated API request
    return 'Downloaded files and pickle stored',200

@app.route('/rec', methods=['POST'])
def rec():
    """Given a user id, return a list of
     recommended course ids."""
    # parse request
    data = request.get_json()
   # user_id=request.args.get('userId')
   # num_recs = request.args.get('numRecs')
    #print("Hi"+str(user_id))
    
    # validate args
    '''
    if user_id is None:
        return 'No user provided.', 400
    if num_recs is None:
        num_recs = DEFAULT_RECS
    '''
    
    uid_int = int(data['userId'])

    nrecs_int = int(data['numRecs'])
    print(uid_int)
        #user = int(input("Enter the user id to whom you want to recommend : "))
        

    '''except:
        return 'User id and number of recs arguments must be integers.', 400
'''
    # get recommended courses
    #rec_list = Recommendations().get_recommendations(uid_int, nrecs_int)

    #if rec_list is None:
        #return 'User Id not found or reccomendations not available : %s' % user_id, 400
    predicted_courses = User_item_score1(uid_int,nrecs_int)
    print(" ")
    print("The Recommendations for User Id : 10")
    print("   ")
    print(predicted_courses)

    json_response = json.dumps({'Courses':predicted_courses})
    
    return json_response, 200


@app.route('/readiness_check', methods=['GET'])
def readiness_check():
    return jsonify(readiness_status='Success'), 200



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)