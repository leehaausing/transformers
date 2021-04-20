# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score

    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn


if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()

    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }

    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }

    def glue_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(labels)
        if task_name == "cola":
            return {"mcc": matthews_corrcoef(labels, preds)}
        elif task_name == "sst-2":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mrpc":
            return acc_and_f1(preds, labels)
        elif task_name == "sts-b":
            return pearson_and_spearman(preds, labels)
        elif task_name == "qqp":
            return acc_and_f1(preds, labels)
        elif task_name == "mnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mnli-mm":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "qnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "rte":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "wnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "hans":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "boolq":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "length_control":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "syntactic_category_control":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "irregular_form_control":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "antonyms_control":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "control_raising_control":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "absolute_token_position_control":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "lexical_content_the_control":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "relative_position_control":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "title_case_control":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "main_verb_control":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "antonyms_absolute_token_position_namb":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "antonyms_length_namb":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "antonyms_lexical_content_the_namb":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "antonyms_relative_position_namb":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "antonyms_title_case_namb":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "control_raising_absolute_token_position_namb":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "control_raising_length_namb":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "control_raising_lexical_content_the_namb":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "control_raising_relative_token_position_namb":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "control_raising_title_case_namb":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "irregular_form_absolute_token_position_namb":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "irregular_form_length_namb":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "irregular_form_lexical_content_the_namb":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "irregular_form_relative_token_position_namb":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "irregular_form_title_case_namb":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "main_verb_absolute_token_position_namb":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "main_verb_length_namb":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "main_verb_lexical_content_the_namb":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "main_verb_relative_token_position_namb":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "main_verb_title_case_namb":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "syntactic_category_absolute_position_namb":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "syntactic_category_length_namb":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "syntactic_category_lexical_content_the_namb":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "syntactic_category_relative_position_namb":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "syntactic_category_title_case_namb":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "antonyms_absolute_token_position_001":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "antonyms_length_001":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "antonyms_lexical_content_the_001":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "antonyms_relative_position_001":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "antonyms_title_case_001":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "control_raising_absolute_token_position_001":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "control_raising_length_001":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "control_raising_lexical_content_the_001":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "control_raising_relative_token_position_001":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "control_raising_title_case_001":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "irregular_form_absolute_token_position_001":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "irregular_form_length_001":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "irregular_form_lexical_content_the_001":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "irregular_form_relative_token_position_001":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "irregular_form_title_case_001":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "main_verb_absolute_token_position_001":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "main_verb_length_001":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "main_verb_lexical_content_the_001":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "main_verb_relative_token_position_001":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "main_verb_title_case_001":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "syntactic_category_absolute_position_001":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "syntactic_category_length_001":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "syntactic_category_lexical_content_the_001":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "syntactic_category_relative_position_001":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "syntactic_category_title_case_001":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "antonyms_absolute_token_position_003":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "antonyms_length_003":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "antonyms_lexical_content_the_003":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "antonyms_relative_position_003":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "antonyms_title_case_003":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "control_raising_absolute_token_position_003":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "control_raising_length_003":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "control_raising_lexical_content_the_003":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "control_raising_relative_token_position_003":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "control_raising_title_case_003":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "irregular_form_absolute_token_position_003":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "irregular_form_length_003":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "irregular_form_lexical_content_the_003":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "irregular_form_relative_token_position_003":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "irregular_form_title_case_003":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "main_verb_absolute_token_position_003":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "main_verb_length_003":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "main_verb_lexical_content_the_003":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "main_verb_relative_token_position_003":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "main_verb_title_case_003":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "syntactic_category_absolute_position_003":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "syntactic_category_length_003":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "syntactic_category_lexical_content_the_003":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "syntactic_category_relative_position_003":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "syntactic_category_title_case_003":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "antonyms_absolute_token_position_01":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "antonyms_length_01":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "antonyms_lexical_content_the_01":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "antonyms_relative_position_01":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "antonyms_title_case_01":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "control_raising_absolute_token_position_01":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "control_raising_length_01":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "control_raising_lexical_content_the_01":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "control_raising_relative_token_position_01":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "control_raising_title_case_01":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "irregular_form_absolute_token_position_01":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "irregular_form_length_01":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "irregular_form_lexical_content_the_01":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "irregular_form_relative_token_position_01":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "irregular_form_title_case_01":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "main_verb_absolute_token_position_01":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "main_verb_length_01":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "main_verb_lexical_content_the_01":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "main_verb_relative_token_position_01":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "main_verb_title_case_01":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "syntactic_category_absolute_position_01":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "syntactic_category_length_01":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "syntactic_category_lexical_content_the_01":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "syntactic_category_relative_position_01":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "syntactic_category_title_case_01":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "test_pair":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "main_verb":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "subject_aux_inversion":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "cogsci_paper":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "subj_aux_annotated":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "main_verb_annotated":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "reflexives_annotated":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)

    def xnli_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(labels)
        if task_name == "xnli":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)
