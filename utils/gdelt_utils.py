"""
gdelt_utils.py
==============
Utilitaires GDELT : connexion DuckDB, whitelist des sources, nettoyage.

DEUX STRATÉGIES DE CHARGEMENT — choisissez selon le contexte :

  ┌─────────────────────────────────────────────────────────────────┐
  │  NOTEBOOK exploratoire  →  build_clean_views()  (lazy)         │
  │    • Un seul scan parquet par cellule                          │
  │    • Predicate pushdown vers le lecteur parquet                │
  │    • Zéro écriture disque intermédiaire                        │
  │    • Idéal quand chaque requête est différente                 │
  ├─────────────────────────────────────────────────────────────────┤
  │  Script de production  →  build_clean_table()  (matérialisé)   │
  │    • Un seul scan total, résultat écrit en mémoire             │
  │    • Toutes les requêtes suivantes lisent la TABLE             │
  │    • Rentable si gkg_clean est lue N≥3 fois (N secteurs)      │
  └─────────────────────────────────────────────────────────────────┘

Usage notebook (GdeltLoader) :
    from gdelt_utils import GdeltLoader
    loader = GdeltLoader(parquet_dir=..., source_map=..., min_words=20)
    con = loader.as_views()          # ← notebook : vues lazys
    con = loader.as_table()          # ← prod : TABLE matérialisée
    con.execute("SELECT * FROM gkg_clean LIMIT 5").df()

Usage script (fonctions autonomes) :
    from gdelt_utils import make_connection, build_whitelist
    from gdelt_utils import build_clean_views, build_clean_table
    con = make_connection(threads=64, memory_gb=150)
    build_whitelist(con, parquet_dir, source_map, whitelist_path="./wl.parquet")
    build_clean_views(con, parquet_dir, source_map,            # notebook
                      min_articles_per_source=30, min_active_years=2,
                      min_words=15, max_words=6500, min_themes=2)
    build_clean_table(con, glob_pattern, source_map,           # prod
                      whitelist_path="./wl.parquet",
                      min_words=15, max_words=6500, min_themes=2)

Vue / Table `gkg_clean` exposée dans les deux cas :
    GKGRECORDID, period (DATE), tone, tone_bin,
    total_themes_count, themes_list (VARCHAR[]), countries_list (VARCHAR[])
"""

import json
import time
from pathlib import Path

import duckdb
import pandas as pd


# ── PARAMÈTRES PAR DÉFAUT ─────────────────────────────────────────────────────
DEFAULTS = dict(
    min_words               = 15,
    max_words               = 6500,
    min_articles_per_source = 30,
    min_active_years        = 2,
    min_themes              = 2,
    threads                 = 4,
    memory_gb               = 32,
    tmp_dir                 = "./duckdb_tmp",
    whitelist_path          = "./valid_sources_whitelist.parquet",
)


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{int(s // 60)}m{int(s % 60):02d}s"


def _load_src_df(source_map: str | Path) -> pd.DataFrame:
    with open(source_map, "r", encoding="utf-8") as f:
        sm = json.load(f)
    return pd.DataFrame({
        "SourceCommonName_ID": [int(k) for k in sm["id_to_source"]],
        "SourceCommonName":    list(sm["id_to_source"].values()),
    })


def _glob_expr(parquet_dir: Path, years: tuple[str, ...]) -> str:
    """Retourne l'expression read_parquet(...) DuckDB pour les années demandées."""
    if not years:
        return f"read_parquet('{parquet_dir / 'gdelt_*.parquet'}')"
    files = []
    for y in years:
        files.extend(sorted(parquet_dir.glob(f"gdelt_{y}*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"Aucun fichier gdelt_<année>*.parquet pour {years} dans {parquet_dir}"
        )
    listing = ", ".join(f"'{f}'" for f in files)
    return f"read_parquet([{listing}])"


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONNEXION
# ══════════════════════════════════════════════════════════════════════════════

def make_connection(
    threads:   int        = DEFAULTS["threads"],
    memory_gb: int        = DEFAULTS["memory_gb"],
    tmp_dir:   str | Path = DEFAULTS["tmp_dir"],
) -> duckdb.DuckDBPyConnection:
    """
    Ouvre et configure une connexion DuckDB.

    Parameters
    ----------
    threads   : nombre de threads alloués à DuckDB
    memory_gb : limite RAM en Go
    tmp_dir   : répertoire pour les fichiers temporaires DuckDB
    """
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA memory_limit='{memory_gb}GB'")
    con.execute(f"PRAGMA temp_directory='{tmp_dir}'")
    con.execute("SET preserve_insertion_order=false")
    return con


# ══════════════════════════════════════════════════════════════════════════════
# 2. WHITELIST DES SOURCES  (calcul unique, mis en cache sur disque)
# ══════════════════════════════════════════════════════════════════════════════

def build_whitelist(
    con:                     duckdb.DuckDBPyConnection,
    parquet_dir:             str | Path,
    source_map:              str | Path,
    whitelist_path:          str | Path = DEFAULTS["whitelist_path"],
    min_articles_per_source: int        = DEFAULTS["min_articles_per_source"],
    min_active_years:        int        = DEFAULTS["min_active_years"],
    force:                   bool       = False,
) -> Path:
    """
    Calcule la whitelist des sources valides et la sauvegarde en parquet.

    Scan unique de toutes les années. Ne recalcule que si le fichier est
    absent ou si force=True. La table `global_valid_sources` est créée dans `con`.

    Parameters
    ----------
    con                     : connexion DuckDB active
    parquet_dir             : dossier contenant les gdelt_*.parquet
    source_map              : chemin vers gdelt_sources_mapping.json
    whitelist_path          : où écrire/lire la whitelist
    min_articles_per_source : articles minimum par source (toutes années)
    min_active_years        : années d'activité distinctes minimum
    force                   : recalcule même si le fichier existe déjà
    """
    whitelist_path = Path(whitelist_path)

    if whitelist_path.exists() and not force:
        n = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{whitelist_path}')"
        ).fetchone()[0]
        print(f"[whitelist] Fichier existant réutilisé — {n:,} sources ({whitelist_path})")
        return whitelist_path

    glob_pattern = str(Path(parquet_dir) / "gdelt_*.parquet")
    con.register("src_map", _load_src_df(source_map))

    print(f"[whitelist] Calcul sur toute la base "
          f"(min_articles={min_articles_per_source}, min_years={min_active_years}) …")
    t0 = time.time()

    con.execute(fr"""
        CREATE OR REPLACE TABLE global_valid_sources AS
        WITH raw_src AS (
            SELECT
                COALESCE(NULLIF(r.SourceCommonName_ID, 0), m.SourceCommonName_ID) AS Src_ID,
                substr(CAST(DATE AS VARCHAR), 1, 4) AS pub_year
            FROM read_parquet('{glob_pattern}') r
            LEFT JOIN src_map m
              ON RTRIM(regexp_extract(r.DocumentIdentifier,
                   'https?://(?:www\.)?([^/?:]+)', 1), '\.') = m.SourceCommonName
            WHERE regexp_matches(CAST(DATE AS VARCHAR), '^\d{{14}}$')
              AND GKGRECORDID != '20210925181500-T1111'
        )
        SELECT Src_ID
        FROM raw_src
        WHERE Src_ID IS NOT NULL
        GROUP BY Src_ID
        HAVING COUNT(*)                 >= {min_articles_per_source}
           AND COUNT(DISTINCT pub_year) >= {min_active_years}
    """)
    con.execute(f"COPY global_valid_sources TO '{whitelist_path}' (FORMAT PARQUET)")

    n = con.execute("SELECT COUNT(*) FROM global_valid_sources").fetchone()[0]
    print(f"[whitelist] ✓ {n:,} sources valides → {whitelist_path}  ({_elapsed(t0)})")
    return whitelist_path


# ══════════════════════════════════════════════════════════════════════════════
# 3a. VUES LAZYS  gkg_clean  ←  NOTEBOOK / requêtes one-shot
# ══════════════════════════════════════════════════════════════════════════════

def build_clean_views(
    con:                     duckdb.DuckDBPyConnection,
    parquet_dir:             str | Path,
    source_map:              str | Path,
    min_articles_per_source: int        = DEFAULTS["min_articles_per_source"],
    min_active_years:        int        = DEFAULTS["min_active_years"],
    min_words:               int        = DEFAULTS["min_words"],
    max_words:               int        = DEFAULTS["max_words"],
    min_themes:              int        = DEFAULTS["min_themes"],
) -> None:
    """
    Crée une unique vue lazy `gkg_clean` sur les parquets GDELT.

    ✔ Recommandé pour les notebooks : chaque requête déclenche un seul scan
    parquet avec predicate pushdown et zéro écriture disque.

    Note : les vues intermédiaires (gkg_src, gkg_words…) n'apportent aucun
    gain de performance — DuckDB compile de toute façon toute la chaîne en
    un seul plan physique. Une vue unique est équivalente et plus lisible.

    Parameters
    ----------
    con                     : connexion DuckDB active
    parquet_dir             : dossier contenant les gdelt_*.parquet
    source_map              : chemin vers gdelt_sources_mapping.json
    min_articles_per_source : articles minimum par source
    min_active_years        : années d'activité distinctes minimum
    min_words / max_words   : bornes sur WordCount
    min_themes              : thèmes minimum par article
    """
    parquet_dir  = Path(parquet_dir)
    glob_pattern = str(parquet_dir / "gdelt_*.parquet")
    con.register("src_map", _load_src_df(source_map))

    t0 = time.time()
    print(f"[views] Création de la vue lazy gkg_clean "
          f"(words={min_words}-{max_words}, "
          f"min_themes={min_themes}, "
          f"src≥{min_articles_per_source}art/≥{min_active_years}ans) …")

    con.execute(fr"""
        CREATE OR REPLACE VIEW gkg_clean AS
        WITH
        -- ── Filtres bon marché pushdown au niveau parquet ─────────────────
        base AS (
            SELECT
                r.* EXCLUDE (SourceCommonName_ID),
                COALESCE(NULLIF(r.SourceCommonName_ID, 0), m.SourceCommonName_ID) AS Src_ID
            FROM read_parquet('{glob_pattern}') r
            LEFT JOIN src_map m
              ON RTRIM(regexp_extract(r.DocumentIdentifier,
                   'https?://(?:www\.)?([^/?:]+)', 1), '.') = m.SourceCommonName
            WHERE regexp_matches(CAST(DATE AS VARCHAR), '^\d{{14}}$')
              AND GKGRECORDID != '20210925181500-T1111'
              AND EnhancedThemes IS NOT NULL AND EnhancedThemes != ''
              AND WordCount BETWEEN {min_words} AND {max_words}
              AND COALESCE(NULLIF(r.SourceCommonName_ID, 0), m.SourceCommonName_ID) IS NOT NULL
        ),
        -- ── Sources avec ≥ N articles ET ≥ N années (agrégation sur base filtrée) ──
        valid_sources AS (
            SELECT Src_ID
            FROM base
            GROUP BY Src_ID
            HAVING COUNT(*)                                          >= {min_articles_per_source}
               AND COUNT(DISTINCT substr(CAST(DATE AS VARCHAR), 1, 4)) >= {min_active_years}
        )
        -- ── Colonnes finales ──────────────────────────────────────────────
        SELECT
            b.GKGRECORDID,
            strptime(substr(CAST(b.DATE AS VARCHAR), 1, 8), '%Y%m%d')::DATE AS period,
            CAST(b.Tone AS DOUBLE)                                             AS tone,
            SIGN(CAST(b.Tone AS DOUBLE))                                       AS tone_bin,
            ARRAY_LENGTH(string_split(b.EnhancedThemes, ';'))               AS total_themes_count,
            list_transform(
                string_split(b.EnhancedThemes, ';'),
                x -> upper(trim(split_part(trim(x), ',', 1)))
            )                                                                  AS themes_list,
            list_filter(
                list_distinct(
                    list_transform(
                        string_split(b.EnhancedLocations, ';'),
                        x -> split_part(x, '#', 3)
                    )
                ),
                c -> c != ''
            )                                                                  AS countries_list
        FROM base b
        INNER JOIN valid_sources v ON b.Src_ID = v.Src_ID
        WHERE ARRAY_LENGTH(string_split(b.EnhancedThemes, ';')) >= {min_themes}
    """)

    print(f"[views] ✓ Vue gkg_clean créée en {_elapsed(t0)}")
    print("  → Aucune donnée lue : chaque requête déclenchera son propre scan.")


# ══════════════════════════════════════════════════════════════════════════════
# 3b. TABLE MATÉRIALISÉE  gkg_clean  ←  SCRIPTS PROD (N secteurs)
# ══════════════════════════════════════════════════════════════════════════════

def build_clean_table(
    con:            duckdb.DuckDBPyConnection,
    glob_pattern:   str,
    source_map:     str | Path,
    whitelist_path: str | Path = DEFAULTS["whitelist_path"],
    min_words:      int        = DEFAULTS["min_words"],
    max_words:      int        = DEFAULTS["max_words"],
    min_themes:     int        = DEFAULTS["min_themes"],
    table_name:     str        = "gkg_clean",
) -> None:
    """
    Matérialise `gkg_clean` (ou `table_name`) dans `con`.

    ✔ Recommandé pour les scripts de production où gkg_clean est lue
    plusieurs fois (une par secteur). Le coût d'un scan unique est amorti.

    Nécessite une whitelist préalablement calculée par build_whitelist().

    Parameters
    ----------
    con            : connexion DuckDB active
    glob_pattern   : expression read_parquet DuckDB (glob ou liste de fichiers)
                     ex: "/data/gdelt/gdelt_2022*.parquet"
    source_map     : chemin vers gdelt_sources_mapping.json
    whitelist_path : parquet produit par build_whitelist()
    min_words      : bornes sur WordCount
    max_words      : bornes sur WordCount
    min_themes     : thèmes minimum par article
    table_name     : nom de la table créée dans DuckDB
    """
    whitelist_path = Path(whitelist_path)
    if not whitelist_path.exists():
        raise FileNotFoundError(
            f"Whitelist introuvable : {whitelist_path}\n"
            "  → Appelez build_whitelist() en premier."
        )

    con.register("src_map", _load_src_df(source_map))

    t0 = time.time()
    print(f"[table] Matérialisation de '{table_name}' "
          f"(words={min_words}-{max_words}, min_themes={min_themes}) …")

    con.execute(fr"""
        CREATE OR REPLACE TABLE {table_name} AS
        WITH raw AS (
            SELECT * FROM {glob_pattern}
            WHERE regexp_matches(CAST(DATE AS VARCHAR), '^\d{{14}}$')
              AND GKGRECORDID != '20210925181500-T1111'
              AND EnhancedThemes IS NOT NULL AND EnhancedThemes != ''
              AND WordCount BETWEEN {min_words} AND {max_words}
        ),
        mapped AS (
            SELECT
                r.* EXCLUDE (SourceCommonName_ID),
                COALESCE(NULLIF(r.SourceCommonName_ID, 0),
                         m.SourceCommonName_ID) AS Src_ID
            FROM raw r
            LEFT JOIN src_map m
              ON RTRIM(regexp_extract(r.DocumentIdentifier,
                   'https?://(?:www\.)?([^/?:]+)', 1), '\.') = m.SourceCommonName
            WHERE COALESCE(NULLIF(r.SourceCommonName_ID, 0),
                           m.SourceCommonName_ID) IS NOT NULL
        )
        SELECT
            m.GKGRECORDID,
            strptime(substr(CAST(m.DATE AS VARCHAR), 1, 8), '%Y%m%d')::DATE AS period,
            CAST(m.Tone AS DOUBLE)                                           AS tone,
            SIGN(CAST(m.Tone AS DOUBLE))                                     AS tone_bin,
            ARRAY_LENGTH(string_split(m.EnhancedThemes, ';'))               AS total_themes_count,
            list_transform(
                string_split(m.EnhancedThemes, ';'),
                x -> upper(trim(split_part(trim(x), ',', 1)))
            )                                                                AS themes_list,
            list_filter(
                list_distinct(
                    list_transform(
                        string_split(m.EnhancedLocations, ';'),
                        x -> split_part(x, '#', 3)
                    )
                ),
                c -> c != ''
            )                                                                AS countries_list
        FROM mapped m
        INNER JOIN read_parquet('{whitelist_path}') wl ON m.Src_ID = wl.Src_ID
        WHERE ARRAY_LENGTH(string_split(m.EnhancedThemes, ';')) >= {min_themes}
    """)

    n = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    if n == 0:
        print(
            f"[table] ⚠ '{table_name}' est vide.\n"
            "  → Whitelist incompatible avec ce sous-ensemble ?\n"
            "  → Régénérez avec build_whitelist(force=True) ou réduisez les seuils."
        )
    else:
        rng = con.execute(
            f"SELECT MIN(period), MAX(period) FROM {table_name}"
        ).fetchone()
        print(f"[table] ✓ '{table_name}' : {n:,} articles "
              f"({rng[0]} → {rng[1]})  [{_elapsed(t0)}]")


# ══════════════════════════════════════════════════════════════════════════════
# 4. CLASSE  GdeltLoader  (interface unifiée notebook + script)
# ══════════════════════════════════════════════════════════════════════════════

class GdeltLoader:
    """
    Interface unifiée pour notebooks et scripts.

    Méthodes de chargement :
        loader.as_views(*years)   → vues lazys    (notebook, requêtes one-shot)
        loader.as_table(*years)   → TABLE         (prod, N lectures de gkg_clean)

    Les deux exposent la même vue/table `gkg_clean` avec les mêmes colonnes.

    Exemple notebook
    ----------------
    loader = GdeltLoader(
        parquet_dir = "/data/gdelt/gdelt_parquet_db",
        source_map  = "/data/gdelt/gdelt_sources_mapping.json",
        threads=4, memory_gb=32,
    )
    con = loader.as_views()           # toutes les années
    con = loader.as_views("2022")     # une seule année
    loader.info()
    loader.sample(10)

    Exemple script prod
    -------------------
    loader = GdeltLoader(parquet_dir=..., source_map=..., threads=64, memory_gb=150)
    loader.ensure_whitelist()         # une seule fois
    con = loader.as_table("2022")     # matérialisé, relu par N secteurs
    """

    def __init__(
        self,
        parquet_dir:             str | Path,
        source_map:              str | Path,
        whitelist_path:          str | Path = DEFAULTS["whitelist_path"],
        min_words:               int        = DEFAULTS["min_words"],
        max_words:               int        = DEFAULTS["max_words"],
        min_themes:              int        = DEFAULTS["min_themes"],
        min_articles_per_source: int        = DEFAULTS["min_articles_per_source"],
        min_active_years:        int        = DEFAULTS["min_active_years"],
        threads:                 int        = DEFAULTS["threads"],
        memory_gb:               int        = DEFAULTS["memory_gb"],
        tmp_dir:                 str | Path = DEFAULTS["tmp_dir"],
    ):
        self.parquet_dir             = Path(parquet_dir)
        self.source_map              = Path(source_map)
        self.whitelist_path          = Path(whitelist_path)
        self.min_words               = min_words
        self.max_words               = max_words
        self.min_themes              = min_themes
        self.min_articles_per_source = min_articles_per_source
        self.min_active_years        = min_active_years
        self.threads                 = threads
        self.memory_gb               = memory_gb
        self.tmp_dir                 = Path(tmp_dir)
        self._con: duckdb.DuckDBPyConnection | None = None

    # ── Whitelist ──────────────────────────────────────────────────────────────

    def ensure_whitelist(self, force: bool = False) -> Path:
        """Calcule la whitelist si absente (ou force=True)."""
        con_tmp = make_connection(self.threads, self.memory_gb, self.tmp_dir)
        try:
            path = build_whitelist(
                con                     = con_tmp,
                parquet_dir             = self.parquet_dir,
                source_map              = self.source_map,
                whitelist_path          = self.whitelist_path,
                min_articles_per_source = self.min_articles_per_source,
                min_active_years        = self.min_active_years,
                force                   = force,
            )
        finally:
            con_tmp.close()
        return path

    def _reset_con(self) -> duckdb.DuckDBPyConnection:
        if self._con is not None:
            try:
                self._con.close()
            except Exception:
                pass
        self._con = make_connection(self.threads, self.memory_gb, self.tmp_dir)
        return self._con

    # ── Mode notebook : vues lazys ─────────────────────────────────────────────

    def as_views(self, *years: str) -> duckdb.DuckDBPyConnection:
        """
        Crée la chaîne de vues lazys gkg_raw → gkg_clean.

        Recommandé pour les notebooks : un seul scan par cellule,
        predicate pushdown, zéro écriture disque.

        Parameters
        ----------
        *years : années à filtrer ("2020", "2021" …).
                 Sans argument → toutes les années.
        """
        con = self._reset_con()

        # Si années spécifiées, on crée gkg_raw sur la liste de fichiers
        # puis les vues restantes s'enchaînent normalement
        if years:
            files = []
            for y in years:
                files.extend(sorted(self.parquet_dir.glob(f"gdelt_{y}*.parquet")))
            if not files:
                raise FileNotFoundError(
                    f"Aucun fichier pour les années {years} dans {self.parquet_dir}"
                )
            listing = ", ".join(f"'{f}'" for f in files)
            con.execute(
                f"CREATE OR REPLACE VIEW gkg_raw AS "
                f"SELECT * FROM read_parquet([{listing}])"
            )
            # On passe parquet_dir=None : gkg_raw est déjà créée
            # Astuce : on surcharge la vue dans build_clean_views
        else:
            glob_pattern = str(self.parquet_dir / "gdelt_*.parquet")
            con.execute(
                f"CREATE OR REPLACE VIEW gkg_raw AS "
                f"SELECT * FROM read_parquet('{glob_pattern}')"
            )

        # Enregistre src_map et crée les vues src→clean
        con.register("src_map", _load_src_df(self.source_map))
        self._build_views_from_raw(con)
        return con

    def _build_views_from_raw(self, con: duckdb.DuckDBPyConnection) -> None:
        """Crée une vue unique gkg_clean à partir de la vue gkg_raw existante."""
        con.execute(fr"""
            CREATE OR REPLACE VIEW gkg_clean AS
            WITH
            base AS (
                SELECT
                    r.* EXCLUDE (SourceCommonName_ID),
                    COALESCE(NULLIF(r.SourceCommonName_ID, 0), m.SourceCommonName_ID) AS Src_ID
                FROM gkg_raw r
                LEFT JOIN src_map m
                  ON RTRIM(regexp_extract(r.DocumentIdentifier,
                       'https?://(?:www\.)?([^/?:]+)', 1), '.') = m.SourceCommonName
                WHERE regexp_matches(CAST(DATE AS VARCHAR), '^\d{{14}}$')
                  AND GKGRECORDID != '20210925181500-T1111'
                  AND EnhancedThemes IS NOT NULL AND EnhancedThemes != ''
                  AND WordCount BETWEEN {self.min_words} AND {self.max_words}
                  AND COALESCE(NULLIF(r.SourceCommonName_ID, 0), m.SourceCommonName_ID) IS NOT NULL
            ),
            valid_sources AS (
                SELECT Src_ID
                FROM base
                GROUP BY Src_ID
                HAVING COUNT(*)                                          >= {self.min_articles_per_source}
                   AND COUNT(DISTINCT substr(CAST(DATE AS VARCHAR), 1, 4)) >= {self.min_active_years}
            )
            SELECT
                b.GKGRECORDID,
                strptime(substr(CAST(b.DATE AS VARCHAR), 1, 8), '%Y%m%d')::DATE AS period,
                CAST(b.Tone AS DOUBLE)                                             AS tone,
                SIGN(CAST(b.Tone AS DOUBLE))                                       AS tone_bin,
                ARRAY_LENGTH(string_split(b.EnhancedThemes, ';'))               AS total_themes_count,
                list_transform(
                    string_split(b.EnhancedThemes, ';'),
                    x -> upper(trim(split_part(trim(x), ',', 1)))
                )                                                                  AS themes_list,
                list_filter(
                    list_distinct(
                        list_transform(
                            string_split(b.EnhancedLocations, ';'),
                            x -> split_part(x, '#', 3)
                        )
                    ),
                    c -> c != ''
                )                                                                  AS countries_list
            FROM base b
            INNER JOIN valid_sources v ON b.Src_ID = v.Src_ID
            WHERE ARRAY_LENGTH(string_split(b.EnhancedThemes, ';')) >= {self.min_themes}
        """)
        print("[views] ✓ Vue unique gkg_clean créée (lazy)")

    # ── Mode prod : TABLE matérialisée ────────────────────────────────────────

    def as_table(self, *years: str) -> duckdb.DuckDBPyConnection:
        """
        Matérialise gkg_clean dans une TABLE DuckDB et retourne la connexion.

        Recommandé pour les scripts de production où gkg_clean est lue
        plusieurs fois (ex : une lecture par secteur).

        Parameters
        ----------
        *years : années à charger. Sans argument → toutes les années.
        """
        if not self.whitelist_path.exists():
            print("[as_table] Whitelist absente — génération automatique …")
            self.ensure_whitelist()

        con = self._reset_con()
        expr = _glob_expr(self.parquet_dir, years)

        build_clean_table(
            con            = con,
            glob_pattern   = expr,
            source_map     = self.source_map,
            whitelist_path = self.whitelist_path,
            min_words      = self.min_words,
            max_words      = self.max_words,
            min_themes     = self.min_themes,
        )
        self._con = con
        return con

    # ── Helpers notebook ─────────────────────────────────────────────────────

    def info(self) -> pd.DataFrame:
        """Statistiques descriptives rapides sur gkg_clean."""
        self._check_loaded()
        return self._con.execute("""
            SELECT
                MIN(period)                                        AS date_min,
                MAX(period)                                        AS date_max,
                COUNT(*)                                           AS n_articles,
                COUNT(DISTINCT period)                             AS n_jours,
                ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT period), 1) AS articles_par_jour,
                ROUND(AVG(tone), 4)                                AS tone_moyen,
                ROUND(AVG(tone_bin), 4)                            AS tone_bin_moyen,
                ROUND(AVG(total_themes_count), 2)                  AS themes_moyen
            FROM gkg_clean
        """).df()

    def daily_counts(self) -> pd.DataFrame:
        """Articles par jour."""
        self._check_loaded()
        return self._con.execute("""
            SELECT period, COUNT(*) AS n_articles, AVG(tone) AS tone_moyen
            FROM gkg_clean GROUP BY 1 ORDER BY 1
        """).df()

    def sample(self, n: int = 5) -> pd.DataFrame:
        """n lignes aléatoires de gkg_clean."""
        self._check_loaded()
        return self._con.execute(
            f"SELECT * FROM gkg_clean USING SAMPLE {n}"
        ).df()

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def _check_loaded(self) -> None:
        if self._con is None:
            raise RuntimeError("Appelez d'abord loader.as_views() ou loader.as_table()")

    def __repr__(self) -> str:
        status = "chargée" if self._con is not None else "non chargée"
        return (
            f"GdeltLoader(\n"
            f"  parquet_dir  = {self.parquet_dir}\n"
            f"  whitelist    = {self.whitelist_path}\n"
            f"  words        = [{self.min_words}, {self.max_words}]  "
            f"min_themes={self.min_themes}\n"
            f"  src_filters  = articles≥{self.min_articles_per_source}  "
            f"years≥{self.min_active_years}\n"
            f"  resources    = {self.threads}t / {self.memory_gb}GB\n"
            f"  gkg_clean    = {status}\n"
            f")"
        )