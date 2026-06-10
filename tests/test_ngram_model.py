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


def test_char_ngram_fuzzy_suffix_recovers_single_typo():
    songs = [CleanedSong(source="song", lines=["不论这世界多糟糕，未来的你会光芒万丈", "而我也曾是你万分之一的光"])]
    model = CharNGramModel.train(songs, order=16)

    prediction = model.predict("不论这世界多糟糕，未来的你会光茫万丈")

    assert prediction is not None
    assert prediction.text == "，而我也曾是你万分之一的光"
    assert prediction.reason == "char_ngram_fuzzy"
    assert prediction.corrected_context == "不论这世界多糟糕，未来的你会光芒万丈"


def test_char_ngram_fuzzy_rejects_ambiguous_continuations():
    songs = [
        CleanedSong(source="song-a", lines=["不论这世界多糟糕，未来的你会光芒万丈", "第一种答案"]),
        CleanedSong(source="song-b", lines=["不论这世界多糟糕，未来的你会光芒万丈", "第二种答案"]),
    ]
    model = CharNGramModel.train(songs, order=16)

    assert model.predict("不论这世界多糟糕，未来的你会光茫万丈") is None


def test_char_ngram_uses_long_context_to_disambiguate_repeated_suffix():
    songs = [
        CleanedSong(
            source="song",
            lines=[
                "你应该忘记了吧",
                "天气晴朗",
                "心里却潮湿的盛夏",
                "小小的房间里你弹着吉他",
                "未来的你会光芒万丈",
                "而我也曾是你万分之一的光",
                "那么闪耀",
                "你应该忘记了吧",
                "天气晴朗",
                "心里却潮湿的盛夏",
                "而我还记得你那天写的歌",
            ],
        )
    ]
    model = CharNGramModel.train(songs, order=32)

    short_prediction = model.predict("心里却潮湿的盛夏")
    long_prediction = model.predict("那么闪耀，你应该忘记了吧，天气晴朗，心里却潮湿的盛夏")

    assert short_prediction is None
    assert long_prediction is not None
    assert long_prediction.text == "，而我还记得你那天写的歌"


def test_char_ngram_supports_short_context_for_non_lyric_tasks():
    songs = [CleanedSong(source="quotes", lines=["三人行，必有我师焉。"])]
    model = CharNGramModel.train(songs, order=8, min_context=2)

    prediction = model.predict("三人行，")

    assert prediction is not None
    assert prediction.text == "必有我师焉"


def test_char_ngram_treats_space_as_semantic_separator():
    songs = [CleanedSong(source="song", lines=["我明白 张开翅膀", "需要有怎样的英勇"])]
    model = CharNGramModel.train(songs, order=12, min_context=2)

    prediction = model.predict("我明白")

    assert prediction is not None
    assert prediction.text == "张开翅膀"
