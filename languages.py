"""Whisper dil kodu -> NLLB-200 dil kodu eşlemesi."""

WHISPER_TO_NLLB = {
    "en": "eng_Latn", "zh": "zho_Hans", "de": "deu_Latn", "es": "spa_Latn",
    "ru": "rus_Cyrl", "ko": "kor_Hang", "fr": "fra_Latn", "ja": "jpn_Jpan",
    "pt": "por_Latn", "tr": "tur_Latn", "pl": "pol_Latn", "ca": "cat_Latn",
    "nl": "nld_Latn", "ar": "arb_Arab", "sv": "swe_Latn", "it": "ita_Latn",
    "id": "ind_Latn", "hi": "hin_Deva", "fi": "fin_Latn", "vi": "vie_Latn",
    "he": "heb_Hebr", "uk": "ukr_Cyrl", "el": "ell_Grek", "ms": "zsm_Latn",
    "cs": "ces_Latn", "ro": "ron_Latn", "da": "dan_Latn", "hu": "hun_Latn",
    "ta": "tam_Taml", "no": "nob_Latn", "th": "tha_Thai", "ur": "urd_Arab",
    "hr": "hrv_Latn", "bg": "bul_Cyrl", "lt": "lit_Latn", "mi": "mri_Latn",
    "ml": "mal_Mlym", "cy": "cym_Latn", "sk": "slk_Latn", "te": "tel_Telu",
    "fa": "pes_Arab", "lv": "lvs_Latn", "bn": "ben_Beng", "sr": "srp_Cyrl",
    "az": "azj_Latn", "sl": "slv_Latn", "kn": "kan_Knda", "et": "est_Latn",
    "mk": "mkd_Cyrl", "eu": "eus_Latn", "is": "isl_Latn", "hy": "hye_Armn",
    "ne": "npi_Deva", "mn": "khk_Cyrl", "bs": "bos_Latn", "kk": "kaz_Cyrl",
    "sq": "als_Latn", "sw": "swh_Latn", "gl": "glg_Latn", "mr": "mar_Deva",
    "pa": "pan_Guru", "si": "sin_Sinh", "km": "khm_Khmr", "sn": "sna_Latn",
    "yo": "yor_Latn", "so": "som_Latn", "af": "afr_Latn", "oc": "oci_Latn",
    "ka": "kat_Geor", "be": "bel_Cyrl", "tg": "tgk_Cyrl", "sd": "snd_Arab",
    "gu": "guj_Gujr", "am": "amh_Ethi", "yi": "ydd_Hebr", "lo": "lao_Laoo",
    "uz": "uzn_Latn", "fo": "fao_Latn", "ht": "hat_Latn", "ps": "pbt_Arab",
    "tk": "tuk_Latn", "nn": "nno_Latn", "mt": "mlt_Latn", "sa": "san_Deva",
    "lb": "ltz_Latn", "my": "mya_Mymr", "bo": "bod_Tibt", "tl": "tgl_Latn",
    "mg": "plt_Latn", "as": "asm_Beng", "tt": "tat_Cyrl", "ln": "lin_Latn",
    "ha": "hau_Latn", "ba": "bak_Cyrl", "jw": "jav_Latn", "su": "sun_Latn",
    "yue": "yue_Hant",
}


def to_nllb(whisper_code: str) -> str:
    return WHISPER_TO_NLLB.get(whisper_code, "eng_Latn")
