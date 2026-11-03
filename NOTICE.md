# 上游来源

模型权重由运行者从 jhu-clsp/mmBERT-base 下载，本仓库不重新分发基础模型权重。

数据准备程序使用 Shitao/MLDR、facebook/xnli、unimelb-nlp/wikiann、rajpurkar/squad 和 hfl/cmrc2018。上游数据和模型的许可及使用条件以各自仓库为准。

基础编码器由 Transformers 加载，参数高效适配使用 PEFT。相关库与方法的原始作者和许可保持不变。
