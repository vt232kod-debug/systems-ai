#!/bin/bash
# Збирає звіти (.docx + .pdf) для всіх лабораторних обох дисциплін,
# для яких уже існує report_content.md. Безпечно запускати повторно.
set -u

PY="$HOME/.venvs/systems-ai/bin/python"
GEN="/Users/alex/Programs/Univercity/Systems AI/tools/build_report.py"
SSHI="/Users/alex/Programs/Univercity/Systems AI"
MAPZ="/Users/alex/Programs/Univercity/Modeling and analysis of software"
TEACHER_SSHI="${TEACHER_SSHI:-Фант М.О.}"
TEACHER_MAPZ="${TEACHER_MAPZ:-Власенко О.В.}"
REPO_SSHI="https://github.com/vt232kod-debug/systems-ai"
REPO_MAPZ="https://github.com/vt232kod-debug/software-modeling"
VARIANT_MAPZ="Варіант предметної області: «Півот» — вебсервіс і браузерне розширення
для вивчення іноземної мови на власному контенті користувача"

build() {  # dir num theme prefix discipline
  local dir="$1" num="$2" theme="$3" prefix="$4" disc="$5"
  local repo_arg=() teacher="$TEACHER_MAPZ"
  if [ "$prefix" = "СШІ" ]; then
    repo_arg=(--repo "$REPO_SSHI")
    teacher="$TEACHER_SSHI"
  else
    repo_arg=(--repo "$REPO_MAPZ" --variant "$VARIANT_MAPZ")
  fi
  if [ ! -f "$dir/report_content.md" ]; then
    echo "— пропуск: немає $dir/report_content.md"
    return
  fi
  echo "→ збираю $prefix-ЛР-$num  ($(basename "$dir"))"
  "$PY" "$GEN" --dir "$dir" --num "$num" --theme "$theme" \
      --prefix "$prefix" --discipline "$disc" --teacher "$teacher" ${repo_arg[@]+"${repo_arg[@]}"} 2>&1 | sed 's/^/   /'
}

D_SSHI="Системи штучного інтелекту"
D_MAPZ="Моделювання та аналіз програмного забезпечення"

echo "=== Системи штучного інтелекту ==="
build "$SSHI/lab03_fuzzy"      3 "МОДЕЛЮВАННЯ ЕЛЕМЕНТІВ НЕЧІТКИХ МНОЖИН
ТА ФОРМУВАННЯ НЕЧІТКИХ ПРАВИЛ"                               "СШІ" "$D_SSHI"
build "$SSHI/lab04_regression" 4 "ЗАДАЧА РЕГРЕСІЇ. ВИДИ РЕГРЕСІЇ"      "СШІ" "$D_SSHI"
build "$SSHI/lab05_ensembles"  5 "ДОСЛІДЖЕННЯ МЕТОДІВ
АНСАМБЛЕВОГО НАВЧАННЯ"                                        "СШІ" "$D_SSHI"
build "$SSHI/lab06_bayes"      6 "НАЇВНИЙ КЛАСИФІКАТОР БАЙЄСА.
ПРОГНОЗУВАННЯ ТА ОБРОБКА ДАНИХ"                               "СШІ" "$D_SSHI"
build "$SSHI/lab07_clustering" 7 "НЕКОНТРОЛЬОВАНЕ НАВЧАННЯ.
КЛАСТЕРИЗАЦІЯ ДАНИХ"                                          "СШІ" "$D_SSHI"
build "$SSHI/lab08_tensorflow" 8 "НАВЧАННЯ НЕЙРОМЕРЕЖ.
TENSORFLOW. KERAS"                                            "СШІ" "$D_SSHI"

echo
echo "=== Моделювання та аналіз програмного забезпечення ==="
build "$MAPZ/lab01_usecase"    1 "ОГЛЯД CASE-ІНСТРУМЕНТІВ ТА ПОБУДОВА
ДІАГРАМИ ВАРІАНТІВ ВИКОРИСТАННЯ"                              "МАПЗ" "$D_MAPZ"
build "$MAPZ/lab02_classes"    2 "ПОБУДОВА ДІАГРАМИ КЛАСІВ
ПРЕДМЕТНОЇ ОБЛАСТІ"                                           "МАПЗ" "$D_MAPZ"
build "$MAPZ/lab03_components" 3 "ПОБУДОВА ДІАГРАМ ОБ'ЄКТІВ,
КОМПОНЕНТІВ ТА РОЗГОРТАННЯ"                                   "МАПЗ" "$D_MAPZ"
build "$MAPZ/lab04_activity"   4 "ПОБУДОВА ДІАГРАМИ ДІЯЛЬНОСТІ
БІЗНЕС-ПРОЦЕСУ"                                               "МАПЗ" "$D_MAPZ"

echo
echo "Готово. Знайдені звіти:"
find "$SSHI" "$MAPZ" -name "*-ЛР-*-ВТ-23-2-Камінський.pdf" 2>/dev/null | sort | sed 's|.*/||'
