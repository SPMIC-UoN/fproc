"""Submit a pipeline as a Slurm array job."""

import argparse
import ast
import importlib
import importlib.util
import os
import re
import shlex
import subprocess
import sys
import tempfile

from fproc._version import __version__
from fproc.options import ArgumentParser as FprocArgumentParser


RESERVED_FPROC_OPTIONS = {
    "--pipeline",
    "--input",
    "--input-subfolder",
    "--output",
    "--output-subfolder",
    "--subjid",
    "--subjidx",
}

BATCH_DEFAULTS = {
    "conda_env": "renal_preproc",
    "input_subfolder": "fsort",
    "output_subfolder": "fproc",
    "partition": "defq",
    "cpus_per_task": 1,
    "time": "24:00:00",
    "mem_per_cpu": "64g",
}

ALWAYS_FPROC_ARGS = ["--overwrite", "--autoskip"]
ARRAY_SPEC_PATTERN = re.compile(
    r"^\d+(?:-\d+(?::[1-9]\d*)?)?(?:,\d+(?:-\d+(?::[1-9]\d*)?)?)*(?:%[1-9]\d*)?$"
)


def _static_pipeline_value(node, names):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        pass

    if isinstance(node, ast.Name):
        return names.get(node.id)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "path"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "os"
    ):
        parts = [_static_pipeline_value(arg, names) for arg in node.args]
        if all(isinstance(part, str) for part in parts):
            return os.path.join(*parts)
    return None


def _load_pipeline(pipeline_arg):
    if os.path.isfile(pipeline_arg):
        pipeline_path = os.path.abspath(pipeline_arg)
        with open(pipeline_path, "r") as pipeline_file:
            source = pipeline_file.read()
        tree = ast.parse(source, filename=pipeline_path)
        batch_value = None
        study_dir = None
        static_names = {}
        pipeline_name = os.path.splitext(os.path.basename(pipeline_path))[0]
        has_add_options = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "add_options"
            for node in tree.body
        )
        for node in tree.body:
            if isinstance(node, ast.Assign):
                names = [
                    target.id
                    for target in node.targets
                    if isinstance(target, ast.Name)
                ]
                if "NAME" in names:
                    value = _static_pipeline_value(node.value, static_names)
                    if value is not None:
                        pipeline_name = value
                        static_names["NAME"] = value
                if "STUDYDIR" in names:
                    study_dir = _static_pipeline_value(node.value, static_names)
                if "BATCH" in names:
                    try:
                        batch_value = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        batch_value = None
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == "NAME":
                    value = _static_pipeline_value(node.value, static_names)
                    if value is not None:
                        pipeline_name = value
                        static_names["NAME"] = value
                if node.target.id == "STUDYDIR":
                    study_dir = _static_pipeline_value(node.value, static_names)
                if node.target.id == "BATCH":
                    try:
                        batch_value = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        batch_value = None
        if batch_value is not None:
            if has_add_options:
                return _import_pipeline_path(pipeline_path)
            attributes = {"NAME": pipeline_name}
            if batch_value is not None:
                attributes["BATCH"] = batch_value
            if study_dir is not None:
                attributes["STUDYDIR"] = study_dir
            return type(
                "PipelineBatchConfig",
                (object,),
                attributes,
            )

        return _import_pipeline_path(pipeline_path)
    return importlib.import_module(pipeline_arg)


def _import_pipeline_path(pipeline_path):
    module_name = "_fproc_batch_" + re.sub(
        r"\W+", "_", os.path.splitext(os.path.basename(pipeline_path))[0]
    )
    spec = importlib.util.spec_from_file_location(module_name, pipeline_path)
    if spec is None or spec.loader is None:
        raise ValueError("Could not load pipeline configuration {}".format(pipeline_path))
    pipeline = importlib.util.module_from_spec(spec)
    pipeline_dir = os.path.dirname(pipeline_path)
    sys.path.insert(0, pipeline_dir)
    try:
        spec.loader.exec_module(pipeline)
    finally:
        sys.path.remove(pipeline_dir)
    return pipeline


def validate_fproc_args(pipeline_arg, pipeline, fproc_args):
    """Reject arguments not accepted by fproc's parser for this pipeline."""
    parser = FprocArgumentParser("fproc", __version__)
    add_options = getattr(pipeline, "add_options", None)
    if add_options is not None:
        add_options(parser)

    parser.parse_args(
        [
            "--pipeline",
            pipeline_arg,
            "--input",
            "input",
            "--output",
            "output",
        ]
        + list(fproc_args)
    )


def _validate_config(config, slurm=True):
    required = ("modules", "output")
    missing = [key for key in required if key not in config]
    if "subjects" not in config and "array" not in config:
        missing.append("subjects or array")
    if missing:
        raise ValueError(
            "Pipeline BATCH configuration is missing: {}".format(", ".join(missing))
        )
    config.setdefault("input", config["output"])
    if "subjects" in config and (
        not isinstance(config["subjects"], int) or config["subjects"] < 1
    ):
        raise ValueError("BATCH subjects must be a positive integer")
    if not isinstance(config["modules"], (list, tuple)):
        raise ValueError("BATCH modules must be a list or tuple")
    if any(
        not isinstance(module, str) or not module.strip()
        for module in config["modules"]
    ):
        raise ValueError("BATCH modules must contain non-empty module names")
    if slurm:
        if not re.match(r"^\d+:\d{2}:\d{2}$", str(config["time"])):
            raise ValueError("BATCH time must be in Slurm format HH:MM:SS")
        hours, minutes, seconds = [int(part) for part in config["time"].split(":")]
        if minutes > 59 or seconds > 59:
            raise ValueError("BATCH time must use minutes and seconds between 00 and 59")


def _directive_value(value):
    value = str(value)
    if "\n" in value or "\r" in value:
        raise ValueError("Slurm directive values cannot contain newlines")
    return value


def render_script(
    pipeline_arg,
    pipeline,
    fproc_args,
    cwd=None,
    subjidx=None,
    batch_overrides=None,
    batch_config=None,
):
    """Render the Slurm script from a pipeline's BATCH configuration."""
    pipeline_config = (
        batch_config if batch_config is not None else getattr(pipeline, "BATCH", None)
    )
    if not isinstance(pipeline_config, dict):
        raise ValueError("Pipeline must define a BATCH dictionary")
    batch_overrides = batch_overrides or {}
    config = dict(BATCH_DEFAULTS)
    config.update(pipeline_config)
    config.update(batch_overrides)
    if "output" in batch_overrides and not os.path.isabs(config["output"]):
        study_dir = getattr(pipeline, "STUDYDIR", None)
        if not study_dir:
            raise ValueError(
                "A relative --output requires the pipeline to define STUDYDIR"
            )
        config["output"] = os.path.join(study_dir, config["output"])
    if "partition" not in pipeline_config and "partition" not in batch_overrides:
        config["partition"] = "ampereq" if config.get("gpu", False) else "defq"
    _validate_config(config)

    array_spec = subjidx
    if array_spec is None:
        array_spec = config.get("array")
    if array_spec is None:
        array_spec = "0-{}".format(config["subjects"] - 1)
    if not ARRAY_SPEC_PATTERN.fullmatch(array_spec):
        raise ValueError("Invalid subject index array specification: {}".format(array_spec))

    cwd = os.path.abspath(cwd or os.getcwd())
    pipeline_name = getattr(pipeline, "NAME", "pipeline")
    log_name = re.sub(r"[^A-Za-z0-9_.-]", "_", pipeline_name)
    job_name = config.get("job_name", "{}_fproc".format(pipeline_name))
    job_name = re.sub(r"[^A-Za-z0-9_.-]", "_", job_name)
    relocate_logs = "log_dir" not in pipeline_config
    log_dir = config.get(
        "log_dir", os.path.join(cwd, "logs", log_name)
    )
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(cwd, log_dir)
    log_dir = os.path.abspath(log_dir)

    directives = [
        "#SBATCH --job-name={}".format(_directive_value(job_name)),
        "#SBATCH --time={}".format(_directive_value(config["time"])),
        "#SBATCH --array={}".format(_directive_value(array_spec)),
        "#SBATCH --output={}/%A_%a.out".format(_directive_value(log_dir)),
        "#SBATCH --error={}/%A_%a.err".format(_directive_value(log_dir)),
        "#SBATCH --nodes={}".format(_directive_value(config.get("nodes", 1))),
        "#SBATCH --ntasks={}".format(_directive_value(config.get("ntasks", 1))),
        "#SBATCH --cpus-per-task={}".format(
            _directive_value(config.get("cpus_per_task", 1))
        ),
        "#SBATCH --mem-per-cpu={}".format(
            _directive_value(config["mem_per_cpu"])
        ),
        "#SBATCH --partition={}".format(_directive_value(config.get("partition", "defq"))),
        "#SBATCH --export=NONE",
    ]
    if str(config.get("partition", "")).startswith("img"):
        directives.append("#SBATCH --qos=img")
    if config.get("gpu", False):
        directives.append(
            "#SBATCH --gres={}".format(
                _directive_value(config.get("gpu_resource", "gpu:1"))
            )
        )

    lines = ["#!/bin/bash -l"] + directives + [
        "",
        "cd {}".format(shlex.quote(cwd)),
    ]
    if relocate_logs:
        quoted_log_dir = shlex.quote(log_dir)
        job_log_dir = quoted_log_dir + '/"${SLURM_ARRAY_JOB_ID}"'
        lines.extend(
            [
                "mkdir -p {}".format(job_log_dir),
                "mv {}/\"${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.out\" "
                "{}".format(
                    quoted_log_dir,
                    job_log_dir + '/"${SLURM_ARRAY_TASK_ID}.out"',
                ),
                "mv {}/\"${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.err\" "
                "{}".format(
                    quoted_log_dir,
                    job_log_dir + '/"${SLURM_ARRAY_TASK_ID}.err"',
                ),
            ]
        )
    modules = ["conda-img"] + [
        module_name
        for module_name in config["modules"]
        if module_name != "conda-img"
    ]
    lines.extend(
        "module load {}".format(shlex.quote(module_name))
        for module_name in modules
    )
    conda_env = config.get("conda_env")
    if conda_env:
        lines.append("source activate {}".format(shlex.quote(conda_env)))
    lines.extend(["", 'echo "Processing subject ${SLURM_ARRAY_TASK_ID}"'])

    command = [
        "fproc",
        "--pipeline",
        pipeline_arg,
        "--input",
        config["input"],
    ]
    if config.get("input_subfolder"):
        command.extend(["--input-subfolder", config["input_subfolder"]])
    command.extend(["--output", config["output"]])
    if config.get("output_subfolder"):
        command.extend(["--output-subfolder", config["output_subfolder"]])
    command.extend(ALWAYS_FPROC_ARGS)
    command.extend(config.get("extra_fproc_args", []))
    command.extend(
        [
            "--subjidx",
            "${SLURM_ARRAY_TASK_ID}",
        ]
    )
    command.extend(fproc_args)

    shell_command = " ".join(
        part if part == "${SLURM_ARRAY_TASK_ID}" else shlex.quote(str(part))
        for part in command
    )
    lines.extend([shell_command, ""])
    return "\n".join(lines), log_dir


def _expand_array_spec(array_spec):
    if not ARRAY_SPEC_PATTERN.fullmatch(array_spec):
        raise ValueError("Invalid subject index array specification: {}".format(array_spec))
    indexes = []
    for item in array_spec.split("%", 1)[0].split(","):
        if "-" not in item:
            indexes.append(int(item))
            continue
        bounds, _, stride = item.partition(":")
        start, end = [int(value) for value in bounds.split("-", 1)]
        step = int(stride) if stride else 1
        if end < start:
            raise ValueError("Subject index ranges must be ascending: {}".format(item))
        indexes.extend(range(start, end + 1, step))
    return indexes


def render_shell_script(
    pipeline_arg,
    pipeline,
    fproc_args,
    cwd=None,
    subjidx=None,
    batch_overrides=None,
):
    """Render a shell script that processes selected subjects serially."""
    pipeline_config = getattr(pipeline, "BATCH", None)
    if not isinstance(pipeline_config, dict):
        raise ValueError("Pipeline must define a BATCH dictionary")
    batch_overrides = batch_overrides or {}
    config = dict(BATCH_DEFAULTS)
    config.update(pipeline_config)
    config.update(batch_overrides)
    if "output" in batch_overrides and not os.path.isabs(config["output"]):
        study_dir = getattr(pipeline, "STUDYDIR", None)
        if not study_dir:
            raise ValueError(
                "A relative --output requires the pipeline to define STUDYDIR"
            )
        config["output"] = os.path.join(study_dir, config["output"])
    _validate_config(config, slurm=False)

    array_spec = subjidx or config.get("array")
    if array_spec is None:
        array_spec = "0-{}".format(config["subjects"] - 1)
    indexes = _expand_array_spec(array_spec)

    cwd = os.path.abspath(cwd or os.getcwd())
    modules = ["conda-img"] + [
        module_name
        for module_name in config["modules"]
        if module_name != "conda-img"
    ]
    lines = ["#!/bin/bash -l", "set -e", "", "cd {}".format(shlex.quote(cwd))]
    lines.extend(
        "module load {}".format(shlex.quote(module_name))
        for module_name in modules
    )
    conda_env = config.get("conda_env")
    if conda_env:
        lines.append("source activate {}".format(shlex.quote(conda_env)))
    lines.extend(
        ["", "for subjidx in {}; do".format(" ".join(map(str, indexes)))]
    )
    lines.append('    echo "Processing subject ${subjidx}"')

    command = [
        "fproc",
        "--pipeline",
        pipeline_arg,
        "--input",
        config["input"],
    ]
    if config.get("input_subfolder"):
        command.extend(["--input-subfolder", config["input_subfolder"]])
    command.extend(["--output", config["output"]])
    if config.get("output_subfolder"):
        command.extend(["--output-subfolder", config["output_subfolder"]])
    command.extend(ALWAYS_FPROC_ARGS)
    command.extend(config.get("extra_fproc_args", []))
    command.extend(["--subjidx", '"${subjidx}"'])
    command.extend(fproc_args)

    shell_command = " ".join(
        part if part == '"${subjidx}"' else shlex.quote(str(part))
        for part in command
    )
    lines.extend(["    {}".format(shell_command), "done", ""])
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Submit an fproc pipeline as a Slurm array job"
    )
    parser.add_argument(
        "--pipeline",
        required=True,
        help="Pipeline module or Python pipeline definition file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated Slurm script without submitting it",
    )
    parser.add_argument(
        "--shell",
        action="store_true",
        help="Print a shell script that processes subjects serially instead of submitting to Slurm",
    )
    parser.add_argument(
        "--subjidx",
        help="Subject indexes to submit (Slurm array syntax, e.g. 1,3-5)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Require a GPU for this job",
    )
    parser.add_argument("--partition", help="Override the Slurm partition")
    parser.add_argument("--time", help="Override the Slurm time limit (HH:MM:SS)")
    parser.add_argument(
        "--mem",
        dest="mem_per_cpu",
        help="Override memory per CPU (e.g. 64g)",
    )
    parser.add_argument(
        "--output",
        help="Override output directory (relative paths are resolved from STUDYDIR)",
    )
    options, fproc_args = parser.parse_known_args(argv)
    if any(arg.split("=", 1)[0] == "--profile" for arg in fproc_args):
        parser.error(
            "--profile is no longer supported; define a separate pipeline BATCH configuration"
        )
    if options.subjidx is not None and not ARRAY_SPEC_PATTERN.fullmatch(
        options.subjidx
    ):
        parser.error("--subjidx must be a valid Slurm array specification")
    if fproc_args and fproc_args[0] == "--":
        fproc_args = fproc_args[1:]
    for arg in fproc_args:
        option = arg.split("=", 1)[0]
        if option in RESERVED_FPROC_OPTIONS:
            parser.error(
                "{} is set by the pipeline's BATCH configuration and cannot be overridden".format(
                    option
                )
            )
    options.fproc_args = fproc_args
    return options


def main(argv=None):
    options = parse_args(argv)
    try:
        pipeline = _load_pipeline(options.pipeline)
        config = getattr(pipeline, "BATCH", None)
        if not isinstance(config, dict):
            raise ValueError("Pipeline must define a BATCH configuration")
        if not options.dry_run:
            validate_fproc_args(
                options.pipeline,
                pipeline,
                ALWAYS_FPROC_ARGS
                + config.get("extra_fproc_args", [])
                + options.fproc_args,
            )
        if options.shell:
            script = render_shell_script(
                options.pipeline,
                pipeline,
                options.fproc_args,
                subjidx=options.subjidx,
                batch_overrides=(
                    {"output": options.output} if options.output is not None else {}
                ),
            )
            print(script, end="")
            return 0
        batch_overrides = {
            name: value
            for name, value in (
                ("partition", options.partition),
                ("time", options.time),
                ("mem_per_cpu", options.mem_per_cpu),
                ("output", options.output),
            )
            if value is not None
        }
        if options.gpu:
            batch_overrides["gpu"] = True
        script, log_dir = render_script(
            options.pipeline,
            pipeline,
            options.fproc_args,
            subjidx=options.subjidx,
            batch_overrides=batch_overrides,
            batch_config=config,
        )
    except (ImportError, OSError, ValueError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        return 1

    if options.dry_run:
        print(script, end="")
        return 0

    try:
        os.makedirs(log_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="fproc-batch-", suffix=".sh", delete=False
        ) as script_file:
            script_file.write(script)
            script_path = script_file.name
        try:
            result = subprocess.run(
                ["sbatch", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
        finally:
            os.unlink(script_path)
    except OSError as error:
        print("ERROR: Could not submit Slurm job: {}".format(error), file=sys.stderr)
        return 1

    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
