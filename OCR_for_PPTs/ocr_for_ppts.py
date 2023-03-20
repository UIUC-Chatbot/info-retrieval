                   ###################################### Q&A ###################################### 
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class OCR_QA:
  def __init__(self, doc_names, docs,seg_doc):
    self.seg_docs = seg_doc
    self.doc_names = doc_names
    self.docs = docs

  def get_slides(self,candidate_docs,doc_names,docs):
    result = []
    for cand_doc in candidate_docs:
      for j in range(0,len(docs)):
        if cand_doc == docs[j]:
          result.append((doc_names[j].split('_')[0],doc_names[j].split('_')[1].split('.')[0] + '.jpg'))
    
    return result


  def get_top_k_articles(self,query, docs, k=2):

    # Initialize a vectorizer that removes English stop words
    vectorizer = TfidfVectorizer(analyzer="word", stop_words='english')

    # Create a corpus of query and documents and convert to TFIDF vectors
    query_and_docs = [query] + docs
    matrix = vectorizer.fit_transform(query_and_docs)

    # Holds our cosine similarity scores
    scores = []

    # The first vector is our query text, so compute the similarity of our query against all document vectors
    for i in range(1, len(query_and_docs)):
      scores.append(cosine_similarity(matrix[0], matrix[i])[0][0])

    # Sort list of scores and return the top k highest scoring documents
    sorted_list = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_doc_indices = [x[0] for x in sorted_list[:k]]
    top_docs = [docs[x] for x in top_doc_indices]
    
    return top_docs

  def get_candidates(self, question, slide_num =4):
    top_candidates = self.get_top_k_articles(question,self.seg_docs,slide_num)
    slides = self.get_slides(top_candidates, self.doc_names, self.docs)
    return slides


# def segment_documents(docs, max_doc_length=450):
#         # List containing full and segmented docs
#         segmented_docs = []
#         for doc in docs:
#             # Split document by spaces to obtain a word count that roughly approximates the token count
#             split_to_words = doc.split(" ")

#             # If the document is longer than our maximum length, split it up into smaller segments and add them to the list 
#             if len(split_to_words) > max_doc_length:
#                 for doc_segment in range(0, len(split_to_words), max_doc_length):
#                     segmented_docs.append( " ".join(split_to_words[doc_segment:doc_segment + max_doc_length]))

#             # If the document is shorter than our maximum length, add it to the list
#             else:
#                 segmented_docs.append(doc)

#         return segmented_docs

# if __name__ == "__main__":
#     DOC_FOLDER = '/home/rsalvi/chatbotai/rohan/info-retrieval/OCR_for_PPTs/documents/content/documents'
#     documents = os.listdir(DOC_FOLDER)
#     docs = []
#     doc_names = []
#     for doc in documents:
#       with open(os.path.join(DOC_FOLDER,doc), 'r') as file:
#         data = file.read().replace('\n', ' ')
#       docs.append(data)
#       doc_names.append(doc)

#     seg_doc = segment_documents(docs)

#     qa = OCR_QA(doc_names,docs,seg_doc)
#     print(qa.get_candidates(question = 'what is unsigned bit?',slide_num = 2))