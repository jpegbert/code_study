https://github.com/youhebuke/dssm-1

数据来自天池大数据比赛，是OPPO手机搜索排序query-title语义匹配的问题。

数据格式： 数据分4列，\t分隔。

| 字段             | 说明                                                         | 数据示例                                  |
| ---------------- | ------------------------------------------------------------ | ----------------------------------------- |
| prefix           | 用户输入（query前缀）                                        | 刘德                                      |
| query_prediction | 根据当前前缀，预测的用户完整需求查询词，最多10条；预测的查询词可能是前缀本身，数字为统计概率 | {“刘德华”: “0.5”, “刘德华的歌”: “0.3”, …} |
| title            | 文章标题                                                     | 刘德华                                    |
| tag              | 文章内容标签                                                 | 百科                                      |
| label            | 是否点击                                                     | 0或1                                      |

为了应用来训练DSSM demo，将prefix和title作为正样，prefix和query_prediction（除title以外）作为负样本。

下载链接：链接: https://pan.baidu.com/s/1Hg2Hubsn3GEuu4gubbHCzw 提取码: 7p3n

本数据仅限用于个人实验，如数据版权问题，请联系[chou.young@qq.com](mailto:chou.young@qq.com) 下架。



下载解压到data文件夹即可，注意修改config.py中配置：file_train, file_vali。
