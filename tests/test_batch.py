import contextlib
import io
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from fproc.batch import (
    BATCH_DEFAULTS,
    _load_pipeline,
    main,
    parse_args,
    render_script,
    render_shell_script,
    validate_fproc_args,
)


class BatchScriptTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = SimpleNamespace(
            NAME="test_pipeline",
            BATCH={
                "modules": ["fsl", "conda-img", "conda"],
                "conda_env": "test-env",
                "input": "/data/input",
                "input_subfolder": "fsort",
                "output": "/data/output",
                "output_subfolder": "fproc",
                "gpu": False,
                "subjects": 12,
                "time": "12:30:00",
                "mem_per_cpu": "8g",
                "extra_fproc_args": ["--custom-option", "custom-value"],
            },
        )

    def test_render_array_script_and_pass_fproc_options(self):
        script, log_dir = render_script(
            "pipelines/test.py",
            self.pipeline,
            ["--force", "--skip", "qa"],
            cwd="/work",
        )

        self.assertIn("#SBATCH --array=0-11", script)
        self.assertIn("#SBATCH --time=12:30:00", script)
        module_loads = [
            line for line in script.splitlines() if line.startswith("module load ")
        ]
        self.assertEqual(
            module_loads,
            [
                "module load conda-img",
                "module load fsl",
                "module load conda",
            ],
        )
        self.assertIn("module load fsl", script)
        self.assertNotIn("#SBATCH --gres=", script)
        self.assertIn("--subjidx ${SLURM_ARRAY_TASK_ID}", script)
        self.assertIn(
            "--overwrite --autoskip --custom-option custom-value "
            "--subjidx ${SLURM_ARRAY_TASK_ID} --force --skip qa",
            script,
        )
        self.assertEqual(log_dir, "/work/logs/test_pipeline")
        self.assertIn("#SBATCH --output=/work/logs/test_pipeline/%A_%a.out", script)
        self.assertIn(
            "mkdir -p /work/logs/test_pipeline/\"${SLURM_ARRAY_JOB_ID}\"", script
        )
        self.assertIn(
            "/work/logs/test_pipeline/\"${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out\"",
            script,
        )
        self.assertIn(
            "/work/logs/test_pipeline/\"${SLURM_ARRAY_JOB_ID}\"/\"${SLURM_ARRAY_TASK_ID}.err\"",
            script,
        )

    def test_render_shell_script_processes_subjects_serially(self):
        pipeline = SimpleNamespace(
            NAME="shell_pipeline",
            BATCH={
                "modules": ["fsl", "conda-img"],
                "input": "/data/input",
                "output": "/data/output",
                "array": "1-5:2,8%4",
            },
        )

        script = render_shell_script(
            "pipelines/test.py",
            pipeline,
            ["--force"],
            cwd="/work",
        )

        self.assertNotIn("#SBATCH", script)
        self.assertIn("module load conda-img\nmodule load fsl", script)
        self.assertIn("for subjidx in 1 3 5 8; do", script)
        self.assertIn('echo "Processing subject ${subjidx}"', script)
        self.assertIn("--subjidx \"${subjidx}\" --force", script)
        self.assertIn("set -e", script)

    def test_shell_mode_prints_script_without_submitting(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as pipeline_file:
            pipeline_file.write(
                "NAME = 'sample'\n"
                "BATCH = {'modules': [], 'input': '/input', 'output': '/output', "
                "'subjects': 3}\n"
            )
            pipeline_file.flush()
            stdout = io.StringIO()
            with mock.patch("fproc.batch.subprocess.run") as run:
                with contextlib.redirect_stdout(stdout):
                    return_code = main(
                        ["--pipeline", pipeline_file.name, "--shell"]
                    )

        script = stdout.getvalue()
        self.assertEqual(return_code, 0)
        self.assertIn("for subjidx in 0 1 2; do", script)
        self.assertNotIn("#SBATCH", script)
        run.assert_not_called()

    def test_default_logs_use_pipeline_name(self):
        self.pipeline.STUDYDIR = "/studies/actual_study"

        script, log_dir = render_script(
            "pipelines/test.py", self.pipeline, [], cwd="/work"
        )

        self.assertEqual(log_dir, "/work/logs/test_pipeline")
        self.assertIn("#SBATCH --error=/work/logs/test_pipeline/%A_%a.err", script)
        self.assertIn(
            'mv /work/logs/test_pipeline/"${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"',
            script,
        )

    def test_explicit_log_dir_is_not_relocated(self):
        self.pipeline.BATCH["log_dir"] = "/custom/logs"

        script, log_dir = render_script(
            "pipelines/test.py", self.pipeline, [], cwd="/work"
        )

        self.assertEqual(log_dir, "/custom/logs")
        self.assertIn("#SBATCH --output=/custom/logs/%A_%a.out", script)
        self.assertNotIn("mkdir -p /custom/logs/", script)

    def test_subjidx_overrides_slurm_array(self):
        script, _ = render_script(
            "pipelines/test.py",
            self.pipeline,
            [],
            cwd="/work",
            subjidx="1,3-5",
        )

        self.assertIn("#SBATCH --array=1,3-5", script)
        self.assertIn("--subjidx ${SLURM_ARRAY_TASK_ID}", script)
        self.assertNotIn("--subjidx 1,3-5", script)

    def test_configured_array_is_used_when_cli_array_is_omitted(self):
        self.pipeline.BATCH["array"] = "58,61"
        self.pipeline.BATCH.pop("subjects")

        script, _ = render_script("pipeline.py", self.pipeline, [], cwd="/work")

        self.assertIn("#SBATCH --array=58,61", script)
        self.assertIn("--subjidx ${SLURM_ARRAY_TASK_ID}", script)

    def test_cli_array_overrides_configured_array(self):
        self.pipeline.BATCH["array"] = "58,61"

        script, _ = render_script(
            "pipeline.py",
            self.pipeline,
            [],
            cwd="/work",
            subjidx="3-4",
        )

        self.assertIn("#SBATCH --array=3-4", script)

    def test_dry_run_uses_subjidx_for_array_and_task_id_for_fproc(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as pipeline_file:
            pipeline_file.write(
                "NAME = 'sample'\n"
                "BATCH = {'modules': [], 'input': '/input', 'output': '/output', "
                "'subjects': 10, 'time': '01:00:00'}\n"
            )
            pipeline_file.flush()
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                return_code = main(
                    [
                        "--pipeline",
                        pipeline_file.name,
                        "--dry-run",
                        "--subjidx",
                        "1,3-5",
                    ]
                )

        script = stdout.getvalue()
        self.assertEqual(return_code, 0)
        self.assertIn("#SBATCH --array=1,3-5", script)
        self.assertIn("--subjidx ${SLURM_ARRAY_TASK_ID}", script)
        self.assertNotIn("--subjidx 1,3-5", script)

    def test_dry_run_batch_options_override_pipeline_configuration(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as pipeline_file:
            pipeline_file.write(
                "NAME = 'sample'\n"
                "BATCH = {'modules': [], 'input': '/input', 'output': '/output', "
                "'gpu': False, 'subjects': 10, 'time': '01:00:00', "
                "'partition': 'default', 'mem_per_cpu': '4g'}\n"
            )
            pipeline_file.flush()
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                return_code = main(
                    [
                        "--pipeline",
                        pipeline_file.name,
                        "--dry-run",
                        "--gpu",
                        "--partition",
                        "accelerated",
                        "--time",
                        "02:30:00",
                        "--mem",
                        "32g",
                    ]
                )

        script = stdout.getvalue()
        self.assertEqual(return_code, 0)
        self.assertIn("#SBATCH --gres=gpu:1", script)
        self.assertIn("#SBATCH --partition=accelerated", script)
        self.assertIn("#SBATCH --time=02:30:00", script)
        self.assertIn("#SBATCH --mem-per-cpu=32g", script)

    def test_dry_run_resolves_relative_output_from_studydir(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as pipeline_file:
            pipeline_file.write(
                "NAME = 'sample'\n"
                "import os\n"
                "STUDYDIR = os.path.join('/data', NAME)\n"
                "BATCH = {'modules': [], 'output': '/data/default', "
                "'subjects': 10, 'time': '01:00:00'}\n"
            )
            pipeline_file.flush()
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                return_code = main(
                    [
                        "--pipeline",
                        pipeline_file.name,
                        "--dry-run",
                        "--output",
                        "alternate",
                    ]
                )

        script = stdout.getvalue()
        self.assertEqual(return_code, 0)
        self.assertIn("--input /data/sample/alternate", script)
        self.assertIn("--output /data/sample/alternate", script)

    def test_dry_run_skips_fproc_argument_validation(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as pipeline_file:
            pipeline_file.write(
                "NAME = 'sample'\n"
                "BATCH = {'modules': [], 'subjects': 1, 'time': '01:00:00', "
                "'extra_fproc_args': ['--pipeline-specific-option']}\n"
            )
            pipeline_file.flush()
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                return_code = main(
                    [
                        "--pipeline",
                        pipeline_file.name,
                        "--dry-run",
                        "--output",
                        "/data/sample",
                    ]
                )

        script = stdout.getvalue()
        self.assertEqual(return_code, 0)
        self.assertIn("--output /data/sample", script)
        self.assertIn("--pipeline-specific-option", script)

    def test_cli_parses_batch_overrides(self):
        options = parse_args(
            [
                "--pipeline",
                "pipelines/test.py",
                "--gpu",
                "--partition",
                "accelerated",
                "--time",
                "02:30:00",
                "--mem",
                "32g",
                "--output",
                "new_output",
            ]
        )

        self.assertTrue(options.gpu)
        self.assertEqual(options.partition, "accelerated")
        self.assertEqual(options.time, "02:30:00")
        self.assertEqual(options.mem_per_cpu, "32g")
        self.assertEqual(options.output, "new_output")
        self.assertEqual(options.fproc_args, [])

    def test_relative_output_override_is_resolved_from_studydir(self):
        self.pipeline.STUDYDIR = "/data/study"

        script, _ = render_script(
            "pipeline.py",
            self.pipeline,
            [],
            cwd="/work",
            batch_overrides={"output": "alternate"},
        )

        self.assertIn("--output /data/study/alternate", script)
        self.assertIn("--input /data/input", script)

    def test_absolute_output_override_is_preserved(self):
        self.pipeline.STUDYDIR = "/data/study"

        script, _ = render_script(
            "pipeline.py",
            self.pipeline,
            [],
            cwd="/work",
            batch_overrides={"output": "/mnt/alternate"},
        )

        self.assertIn("--output /mnt/alternate", script)

    def test_output_override_updates_defaulted_input_directory(self):
        self.pipeline.BATCH.pop("input")
        self.pipeline.STUDYDIR = "/data/study"

        script, _ = render_script(
            "pipeline.py",
            self.pipeline,
            [],
            cwd="/work",
            batch_overrides={"output": "alternate"},
        )

        self.assertIn("--input /data/study/alternate", script)
        self.assertIn("--output /data/study/alternate", script)

    def test_gpu_requirement_adds_gpu_request(self):
        self.pipeline.BATCH["gpu"] = True

        script, _ = render_script("pipeline.py", self.pipeline, [], cwd="/work")

        self.assertIn("#SBATCH --gres=gpu:1", script)

    def test_gpu_defaults_partition_to_ampereq(self):
        self.pipeline.BATCH["gpu"] = True

        script, _ = render_script("pipeline.py", self.pipeline, [], cwd="/work")

        self.assertIn("#SBATCH --partition=ampereq", script)

    def test_gpu_cli_override_defaults_partition_to_ampereq(self):
        script, _ = render_script(
            "pipeline.py",
            self.pipeline,
            [],
            cwd="/work",
            batch_overrides={"gpu": True},
        )

        self.assertIn("#SBATCH --partition=ampereq", script)

    def test_explicit_partition_is_preserved_with_gpu(self):
        self.pipeline.BATCH["gpu"] = True
        self.pipeline.BATCH["partition"] = "customq"

        script, _ = render_script("pipeline.py", self.pipeline, [], cwd="/work")

        self.assertIn("#SBATCH --partition=customq", script)

    def test_img_partition_adds_img_qos(self):
        self.pipeline.BATCH["partition"] = "imgq"

        script, _ = render_script("pipeline.py", self.pipeline, [], cwd="/work")

        self.assertIn("#SBATCH --partition=imgq", script)
        self.assertIn("#SBATCH --qos=img", script)

    def test_non_img_partition_does_not_add_img_qos(self):
        self.pipeline.BATCH["partition"] = "ampereq"

        script, _ = render_script("pipeline.py", self.pipeline, [], cwd="/work")

        self.assertNotIn("#SBATCH --qos=img", script)

    def test_defaults_are_used_when_pipeline_does_not_override_them(self):
        self.pipeline.BATCH.pop("conda_env")
        self.pipeline.BATCH.pop("input_subfolder")
        self.pipeline.BATCH.pop("output_subfolder")
        self.pipeline.BATCH.pop("partition", None)
        self.pipeline.BATCH.pop("cpus_per_task", None)
        self.pipeline.BATCH.pop("time")
        self.pipeline.BATCH.pop("mem_per_cpu")
        self.pipeline.BATCH.pop("extra_fproc_args")

        script, _ = render_script("pipeline.py", self.pipeline, [], cwd="/work")

        self.assertIn("#SBATCH --partition={}".format(BATCH_DEFAULTS["partition"]), script)
        self.assertIn("#SBATCH --time=24:00:00", script)
        self.assertIn("#SBATCH --mem-per-cpu=64g", script)
        self.assertIn(
            "#SBATCH --cpus-per-task={}".format(BATCH_DEFAULTS["cpus_per_task"]),
            script,
        )
        self.assertIn("--input-subfolder fsort", script)
        self.assertIn("--output-subfolder fproc", script)
        self.assertIn("source activate renal_preproc", script)
        self.assertIn("--overwrite --autoskip --subjidx", script)

    def test_input_defaults_to_output_directory(self):
        self.pipeline.BATCH.pop("input")

        script, _ = render_script("pipeline.py", self.pipeline, [], cwd="/work")

        self.assertIn("--input /data/output", script)

    def test_pipeline_can_override_batch_defaults(self):
        self.pipeline.BATCH.update(
            {
                "input_subfolder": "custom_input",
                "output_subfolder": "custom_output",
                "partition": "customq",
                "cpus_per_task": 4,
                "extra_fproc_args": ["--pipeline-specific", "value"],
            }
        )

        script, _ = render_script("pipeline.py", self.pipeline, [], cwd="/work")

        self.assertIn("#SBATCH --partition=customq", script)
        self.assertIn("#SBATCH --cpus-per-task=4", script)
        self.assertIn("--input-subfolder custom_input", script)
        self.assertIn("--output-subfolder custom_output", script)
        self.assertIn(
            "--overwrite --autoskip --pipeline-specific value --subjidx",
            script,
        )

    def test_cli_passes_unknown_fproc_options_through(self):
        options = parse_args(
            ["--pipeline", "pipelines/test.py", "--force", "--skip", "qa"]
        )

        self.assertEqual(options.fproc_args, ["--force", "--skip", "qa"])

    def test_cli_parses_subjidx_array_spec(self):
        options = parse_args(
            [
                "--pipeline",
                "pipelines/test.py",
                "--subjidx",
                "1,3-5",
                "--skip",
                "qa",
            ]
        )

        self.assertEqual(options.subjidx, "1,3-5")
        self.assertEqual(options.fproc_args, ["--skip", "qa"])

    def test_cli_rejects_invalid_subjidx_array_spec(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--pipeline", "pipeline.py", "--subjidx", "1;echo bad"])

    def test_fproc_args_are_checked_against_fproc_parser(self):
        validate_fproc_args(
            "pipeline.py",
            self.pipeline,
            ["--overwrite", "--autoskip", "--force=qa", "--skip=something"],
        )

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                validate_fproc_args(
                    "pipeline.py",
                    self.pipeline,
                    ["--overwrite", "--subjix=3-40,50,51"],
                )

    def test_pipeline_specific_options_are_validated(self):
        def add_options(parser):
            parser.add_argument("--pipeline-option")

        pipeline = SimpleNamespace(
            BATCH=self.pipeline.BATCH,
            add_options=add_options,
        )
        validate_fproc_args("pipeline.py", pipeline, ["--pipeline-option", "value"])

    def test_cli_rejects_overrides_for_array_and_pipeline_paths(self):
        for option in ("--subjidx", "--subjid", "--input"):
            with self.subTest(option=option):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args(["--pipeline", "pipeline.py", option, "override"])

    def test_invalid_time_is_rejected(self):
        self.pipeline.BATCH["time"] = "12:90:00"

        with self.assertRaises(ValueError):
            render_script("pipeline.py", self.pipeline, [], cwd="/work")

    def test_pipeline_batch_metadata_is_loaded_without_importing_pipeline(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as pipeline_file:
            pipeline_file.write(
                "import missing_optional_pipeline_dependency\n"
                "NAME = 'sample'\n"
                "BATCH = {'modules': [], 'input': '/input', 'output': '/output', "
                "'subjects': 1, 'time': '01:00:00'}\n"
            )
            pipeline_file.flush()

            pipeline = _load_pipeline(pipeline_file.name)

        self.assertEqual(pipeline.NAME, "sample")
        self.assertEqual(pipeline.BATCH["input"], "/input")

    def test_cli_rejects_removed_profile_option(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(
                    ["--pipeline", "pipeline.py", "--profile", "special", "--dry-run"]
                )

    def test_submission_uses_sbatch_script_and_returns_confirmation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline_path = os.path.join(temp_dir, "pipeline.py")
            with open(pipeline_path, "w") as pipeline_file:
                pipeline_file.write(
                    "NAME = 'sample'\n"
                    "BATCH = {'modules': [], 'input': '/input', "
                    "'output': '/output', 'subjects': 1, 'time': '01:00:00', "
                    "'log_dir': %r}\n" % os.path.join(temp_dir, "logs")
                )
            result = SimpleNamespace(
                returncode=0,
                stdout="Submitted batch job 123\n",
                stderr="",
            )
            submitted_scripts = []

            def capture_submission(command, **kwargs):
                with open(command[1]) as submitted_script:
                    submitted_scripts.append(submitted_script.read())
                return result

            stdout = io.StringIO()
            with mock.patch(
                "fproc.batch.subprocess.run", side_effect=capture_submission
            ) as run:
                with contextlib.redirect_stdout(stdout):
                    return_code = main(["--pipeline", pipeline_path])

            self.assertEqual(return_code, 0)
            self.assertIn("Submitted batch job 123", stdout.getvalue())
            command = run.call_args[0][0]
            self.assertEqual(command[0], "sbatch")
            script_text = submitted_scripts[0]
            self.assertIn("--subjidx ${SLURM_ARRAY_TASK_ID}", script_text)


if __name__ == "__main__":
    unittest.main()
