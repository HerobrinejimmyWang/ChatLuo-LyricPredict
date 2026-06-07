from lyricpredict.cleaner import CleanedSong
from lyricpredict.ngram_model import CharNGramModel


def test_char_ngram_predicts_until_first_terminator():
    songs = [CleanedSong(source="song", lines=["将故事传颂吧", "风携它远追", "你脸颊热泪"])]
    model = CharNGramModel.train(songs, order=8)

    prediction = model.predict("将故事传颂吧，风携它远追")

    assert prediction is not None
    assert prediction.text == "，你脸颊热泪"


def test_char_ngram_rejects_ambiguous_repeated_context():
    songs = [CleanedSong(source="song", lines=["又一个夜晚 不平凡你已经不算简单", "左边答案", "又一个夜晚 不平凡你已经不算简单", "右边答案"])]
    model = CharNGramModel.train(songs, order=16)

    assert model.predict("又一个夜晚 不平凡你已经不算简单") is None


def test_char_ngram_allows_repeated_context_with_same_continuation():
    songs = [CleanedSong(source="song", lines=["念往昔，我急旋慢转你抚琴低吟", "到如今，重唱此曲却已无你", "念往昔，我急旋慢转你抚琴低吟", "到如今，重唱此曲却已无你"])]
    model = CharNGramModel.train(songs, order=16)

    prediction = model.predict("念往昔，我急旋慢转你抚琴低吟")

    assert prediction is not None
    assert prediction.text == "，到如今"


def test_char_ngram_rejects_long_context_when_only_short_suffix_matches():
    songs = [CleanedSong(source="song", lines=["这天上地下唯一的脚步", "愿君啊 千门万户总有归处"])]
    model = CharNGramModel.train(songs, order=16)

    assert model.predict("有情人自古，不在乎天地之间有几多险阻，这天上地下唯一的脚步") is None
