from gensim.models import TfidfModel
import pandas as pd
import numpy as np
from pprint import pprint
from gensim.corpora import Dictionary
from functools import reduce
import collections


'''
先构建物品画像和用户画像建立，然后基于用户画像和物品画像 实现基于内容的推荐

物品画像
- 利用tags.csv中每部电影的标签作为电影的候选关键词
- 利用TF·IDF计算每部电影的标签的tfidf值，选取TOP-N个关键词作为电影画像标签
- 并将电影的分类词直接作为每部电影的画像标签

用户画像
1. 提取用户观看列表
2. 根据观看列表和物品画像为用户匹配关键词，并统计词频
3. 根据词频排序，最多保留TOP-k个词，这里K设为100，作为用户的标签

基于内容的推荐（内容其实就是从用户和物品中提取的标签）
'''


def get_movie_dataset():
    # 加载基于所有电影的标签
    # all-tags.csv来自ml-latest数据集中
    # 由于ml-latest-small中标签数据太多，因此借助其来扩充
    _tags = pd.read_csv("../data/ml-latest-small/all-tags.csv", usecols=range(1, 3)).dropna()
    tags = _tags.groupby("movieId").agg(list)

    # 加载电影列表数据集
    movies = pd.read_csv("../data/ml-latest-small/movies.csv", index_col="movieId")
    # 将类别词分开
    movies["genres"] = movies["genres"].apply(lambda x: x.split("|"))
    # 为每部电影匹配对应的标签数据，如果没有将会是NAN
    movies_index = set(movies.index) & set(tags.index)
    new_tags = tags.loc[list(movies_index)]
    ret = movies.join(new_tags)

    # 构建电影数据集，包含电影Id、电影名称、类别、标签四个字段
    # 如果电影没有标签数据，那么就替换为空列表
    # map(fun,可迭代对象)
    movie_dataset = pd.DataFrame(
        map(
            lambda x: (x[0], x[1], x[2], x[2]+x[3]) if x[3] is not np.nan else (x[0], x[1], x[2], []), ret.itertuples())
        , columns=["movieId", "title", "genres", "tags"]
    )

    movie_dataset.set_index("movieId", inplace=True)
    return movie_dataset


def create_movie_profile0(movie_dataset):
    '''
    初级版电影画像
    使用tfidf，分析提取topn关键词
    :param movie_dataset:
    :return:
    '''
    dataset = movie_dataset["tags"].values

    # 根据数据集建立词袋，并统计词频，将所有词放入一个词典，使用索引进行获取
    dct = Dictionary(dataset)
    # 根据将每条数据，返回对应的词索引和词频
    corpus = [dct.doc2bow(line) for line in dataset]
    # 训练TF-IDF模型，即计算TF-IDF值
    model = TfidfModel(corpus)

    movie_profile = {}
    for i, mid in enumerate(movie_dataset.index):
        # 根据每条数据返回，向量
        vector = model[corpus[i]]
        # 按照TF-IDF值得到top-n的关键词
        movie_tags = sorted(vector, key=lambda x: x[1], reverse=True)[:30]
        # 根据关键词提取对应的名称
        movie_profile[mid] = dict(map(lambda x: (dct[x[0]], x[1]), movie_tags))

    return movie_profile


def create_movie_profile(movie_dataset):
    '''
    使用tfidf，分析提取topn关键词
    :param movie_dataset:
    :return:
    '''
    dataset = movie_dataset["tags"].values

    # 根据数据集建立词袋，并统计词频，将所有词放入一个词典，使用索引进行获取
    dct = Dictionary(dataset)
    # 根据将每条数据，返回对应的词索引和词频
    corpus = [dct.doc2bow(line) for line in dataset]
    # 训练TF-IDF模型，即计算TF-IDF值
    model = TfidfModel(corpus)

    _movie_profile = []
    for i, data in enumerate(movie_dataset.itertuples()):
        mid = data[0]
        title = data[1]
        genres = data[2]
        vector = model[corpus[i]]
        movie_tags = sorted(vector, key=lambda x: x[1], reverse=True)[:30]
        topN_tags_weights = dict(map(lambda x: (dct[x[0]], x[1]), movie_tags))
        # 将类别词的添加进去，并设置权重值为1.0
        for g in genres:
            topN_tags_weights[g] = 1.0
        topN_tags = [i[0] for i in topN_tags_weights.items()]
        _movie_profile.append((mid, title, topN_tags, topN_tags_weights))

    movie_profile = pd.DataFrame(_movie_profile, columns=["movieId", "title", "profile", "weights"])
    movie_profile.set_index("movieId", inplace=True)
    return movie_profile


def create_inverted_table(movie_profile):
    """
    建立tag-物品的倒排索引
    :param movie_profile: 电影画像
    :return:
    """
    inverted_table = {}
    for mid, weights in movie_profile["weights"].iteritems():
        for tag, weight in weights.items():
            # 到inverted_table dict 用tag作为Key去取值 如果取不到就返回[]
            _ = inverted_table.get(tag, [])
            _.append((mid, weight))
            inverted_table.setdefault(tag, _)
    return inverted_table


def create_user_profile():
    """
    user profile画像建立：
    1. 提取用户观看列表
    2. 根据观看列表和物品画像为用户匹配关键词，并统计词频
    3. 根据词频排序，最多保留TOP-k个词，这里K设为100，作为用户的标签
    :return: user_profile
    """
    watch_record = pd.read_csv("../data/ml-latest-small/ratings.csv", usecols=range(2), dtype={"userId": np.int32, "movieId": np.int32})
    watch_record = watch_record.groupby("userId").agg(list)
    # print(watch_record)

    movie_dataset = get_movie_dataset()
    movie_profile = create_movie_profile(movie_dataset)

    user_profile = {}
    for uid, mids in watch_record.itertuples():
        record_movie_profile = movie_profile.loc[list(mids)]
        counter = collections.Counter(reduce(lambda x, y: list(x)+list(y), record_movie_profile["profile"].values))
        # 兴趣词
        interest_words = counter.most_common(50)
        maxcount = interest_words[0][1]
        interest_words = [(w, round(c/maxcount, 4)) for w, c in interest_words]
        user_profile[uid] = interest_words

    return user_profile


movie_dataset = get_movie_dataset()
print(movie_dataset)

# 物品画像
movie_profile = create_movie_profile(movie_dataset)
pprint(movie_profile)

# 倒排索引
inverted_table = create_inverted_table(movie_profile)
pprint(inverted_table)

# 用户画像
user_profile = create_user_profile()
pprint(user_profile)

# 用户评分数据
watch_record = pd.read_csv("../data/ml-latest-small/ratings.csv", usecols=range(2), dtype={"userId": np.int32, "movieId": np.int32})
watch_record = watch_record.groupby("userId").agg(list)

# 推荐
for uid, interest_words in user_profile.items():
    result_table = {}  # 电影id:[0.2,0.5,0.7]
    for interest_word, interest_weight in interest_words:
        related_movies = inverted_table[interest_word]
        for mid, related_weight in related_movies:
            _ = result_table.get(mid, [])
            _.append(interest_weight)  # 只考虑用户的兴趣程度
            # _.append(related_weight)    # 只考虑兴趣词与电影的关联程度
            # _.append(interest_weight*related_weight)    # 二者都考虑
            result_table.setdefault(mid, _)

    rs_result = map(lambda x: (x[0], sum(x[1])), result_table.items())
    rs_result = sorted(rs_result, key=lambda x: x[1], reverse=True)[:100]
    print(uid)
    pprint(rs_result)
    break

    # 历史数据  ==>  历史兴趣程度 ==>  历史推荐结果       离线推荐    离线计算
    # 在线推荐 ===>    娱乐(王思聪)   ===>   我 ==>  王思聪 100%
    # 近线：最近1天、3天、7天           实时计算

