Word Embeddings and SVD:

(a)
The following steps were taken to achieve the 100-dimensional embeddings:\\
First, all words in the NLTK brown corpus were iterated through. From these, all words that were also in the 'stopwords' corpus of NLTK were filtered out. Also filtered out were words that were in the following list:

. , ? ! #  / * `` '' ; -- one, would, ) (, said, :, new, could, time, two, may, first, like, man, even, made, also, many, must, af, back, years, much, way, well, people, mr., little, state, good, make, world, still, see, men, work, long, get, life, never, day, another, know, last, us, might, great, old, year, come, since, go, came, right, used, three, take, states, use, house, without, place

Additionally, all words containing a dollar sign were removed. The words were then converted to lowercase, and the frequency of all remaining words in the corpus were recorded.
The top 5000 most frequent words were considered to be the vocabulary list V. And the top 1000 most frequent words were considered to be the context words, C.
The value of n(w,c) was calculated, the number of times a context word appeared within a window for each word in V. Using this, the probabilities P(c) and P(c$|$w) were calculated. Then, a 5000x1000 matrix was calculated having all values of $\Phi$, positive mutual information.\\\\
Then, this $\Phi$ matrix was used to calculate the 100-dimesion embeddings, using the sklearndecomposition TruncatedSVD package.

(b)
25 randomly picked vocabulary words and their nearest neighbors in the 100-dimension space are:\\
WORD:    school\\
NEAREST NEIGHBOR:    college\\
=====================================================\\
WORD:    water\\
NEAREST NEIGHBOR:    gas\\
=====================================================\\
WORD:    system\\
NEAREST NEIGHBOR:    group\\
=====================================================\\
WORD:    government\\
NEAREST NEIGHBOR:    public\\
=====================================================\\
WORD:    eyes\\
NEAREST NEIGHBOR:    face\\
=====================================================\\
WORD:    national\\
NEAREST NEIGHBOR:    political\\
=====================================================\\
WORD:    children\\
NEAREST NEIGHBOR:    child\\
=====================================================\\
WORD:    church\\
NEAREST NEIGHBOR:    local\\
=====================================================\\
WORD:    power\\
NEAREST NEIGHBOR:    personal\\
=====================================================\\
WORD:    family\\
NEAREST NEIGHBOR:    group\\
=====================================================\\
WORD:    mind\\
NEAREST NEIGHBOR:    head\\
=====================================================\\
WORD:    country\\
NEAREST NEIGHBOR:    area\\
=====================================================\\
WORD:    service\\
NEAREST NEIGHBOR:    program\\
=====================================================\\
WORD:    god\\
NEAREST NEIGHBOR:    champion\\
=====================================================\\
WORD:    certain\\
NEAREST NEIGHBOR:    different\\
=====================================================\\
WORD:    law\\
NEAREST NEIGHBOR:    laws\\
=====================================================\\
WORD:    human\\
NEAREST NEIGHBOR:    man's\\
=====================================================\\
WORD:    company\\
NEAREST NEIGHBOR:    name\\
=====================================================\\
WORD:    local\\
NEAREST NEIGHBOR:    business\\
=====================================================\\
WORD:    history\\
NEAREST NEIGHBOR:    experience\\
=====================================================\\
WORD:    action\\
NEAREST NEIGHBOR:    upon\\
=====================================================\\
WORD:    feet\\
NEAREST NEIGHBOR:    acres\\
=====================================================\\
WORD:    death\\
NEAREST NEIGHBOR:    upon\\
=====================================================\\
WORD:    experience\\
NEAREST NEIGHBOR:    history\\
=====================================================\\
WORD:    body\\
NEAREST NEIGHBOR:    fat\\
=====================================================\\

Yes, these results make sense since each of the 25 words that were considered has a close neighbor which is either a word that is related to it, or a word that is often used in similar contexts.

(c)
Ward's method was used to calculate the clusters on the 100-dimensional embeddings. This uses the variance minimization algorithm with squared Euclidean distances. The hierarchy.linkage and hierarchy.fcluster methods were used to achieve this from the scipy.cluster library.\\
Some of the best clusters are:
CLUSTER 1:
'say', 'think', 'find', 'give', 'tell', 'keep', 'run', 'leave', 'play', 'believe', 'call', 'read', 'bring', 'talk', 'start', 'remember', 'ask', 'stop', 'remain', 'develop', 'build', 'buy', 'escape', 'explain', 'kill', 'eat', 'lose', 'avoid', 'push'\\\\
This cluster seems to only have verb words.

CLUSTER 2:
'school', 'public', 'business', 'church', 'members', 'local', 'college', 'community', 'university', 'students', 'medical', 'county', 'district', 'student', 'faculty', 'attend'

CLUSTER 3:
'government', 'law', 'federal', 'policy', 'support', 'nations', 'issue', 'science', 'fiscal', 'principle', 'authority', 'civil', 'laws', 'affairs', 'interests', 'politics', 'policies', 'intellectual', 'exist', 'identity', 'administrative', 'conviction', 'attitudes', 'grant', 'encourage', 'psychological', 'participation', 'legislative', 'consequences', 'judgments', 'climate'

CLUSTER 4:
'days', 'least', 'times', 'feet', 'minutes', 'months', 'hours', 'miles', 'weeks', 'dollars', 'inches', 'pounds', 'seconds'
