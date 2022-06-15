

from Levenshtein import distance as lev
import pandas as pd
import numpy as np 

def wisdomain_prep(data) :
    
    data = data[['번호','명칭','요약', '출원인대표명', '출원인국가',
                 '국제특허분류', '공통특허분류', '미국특허분류', '출원일','INPADOC 패밀리',
                 '독립 청구항수', 
                 '전체 청구항수', '대표 청구항', '전체 청구항', '권리 현황', '최종 상태',
                 '자국인용특허', '외국인용특허', '자국인용횟수', '자국피인용횟수' ,'INPADOC패밀리수',
                 'INPADOC패밀리국가수', '발명자수', '소유권이전여부','file_name']]

    data.columns = ['pt_id', 'title', 'abstract', 'applicant', 'country',
                    'IPC', 'CPC', 'USPC', 'application_date', 'family_pat',
                    'ind_claims_cnt',
                    'total_claims_cnt','claims_rep','claims', 'right_status', 'final_status',
                    'cited_pat_in', 'cited_pat_out', 'forward_cited_in_cnt', 'backward_cited_in_cnt', 'family_cnt', 
                    'family_country_cnt','inventor_cnt','transfered','file_name']    
    
    data = data.dropna(subset = ['application_date']).reset_index(drop = 1)
    data['application_year'] = data['application_date'].apply(lambda x : int(x.split('.')[0]))
    data['TA'] = data['title'] + ' ' + data['abstract']
    data['TAF'] = data['title'] + ' ' + data['abstract'] + ' ' + data['claims_rep']
    
    # data['TAF'] = data.apply(lambda x: x.TA +' '+ x.claims_rep if str(x.claims_rep) != 'nan' else x.TA, axis= 1)
    
    # 1. 패밀리 특허 제거
    # for idx, row in data.iterrows() :
        
    #     code = row['pt_id'][0:2]
    #     fam_list = row['family_pat'].split(', ')
    #     fam_list = [i for i in fam_list if i[0:2] == code]
    #     print(fam_list)
    #     for idx_, row_ in data.iterrows() :
    #         if idx != idx_ :
    #             pt_id_ = row['pt_id']
    #             if pt_id_ in fam_list :
    #                 data = data.drop(idx_)
    #                 print(str(idx)+'의 패밀리특허 '+str(idx_)+'제거')
    
    # data = data.reset_index(drop = 1)
    
    # 2. 유사특허 필터링
    # for idx, row in data.iterrows() :
        
    #     text = row['TA'][0:100]
        
    #     for idx_, row_ in data.iterrows() :
    #         if idx != idx_ :
    #             text_ = row_['TA'][0:100]
    #             if lev(text, text_) <= 5 :
    #                 data = data.drop(idx_)
    #                 print(str(idx)+'과 유사한 '+str(idx_)+'제거')
    
    
    return(data)

    data = data.reset_index(drop = 1)
    

    
if __name__ == '__main__':
    
    import os
    import sys
    import re
    import pandas as pd
    import numpy as np     
    import pickle
    from datetime import datetime
    from datetime import timezone
    import spacy

    directory = os.path.dirname(os.path.abspath(__file__))
    directory = directory.replace("\\", "/") # window
    os.chdir(directory)    
    sys.path.append(directory+'/submodule')
    
    directory = 'D:/OneDrive - SNU/db/patent/Wisdomain/'
    file_name = 'blockchain.csv'
    
    data = pd.read_csv(directory + file_name, skiprows=4)
    data['file_name'] = file_name
    data = wisdomain_prep(data)
    