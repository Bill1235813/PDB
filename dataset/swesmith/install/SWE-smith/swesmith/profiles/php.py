import os
import re
import shutil

from dataclasses import dataclass, field
from swebench.harness.constants import (
    FAIL_TO_PASS,
    PASS_TO_PASS,
    KEY_INSTANCE_ID,
    TestStatus,
)
from swesmith.constants import ENV_NAME
from swesmith.profiles.base import RepoProfile, registry


@dataclass
class PhpProfile(RepoProfile):
    """
    Profile for PHP repositories.
    """

    org_dh: str = "swebench"
    test_cmd: str = "vendor/bin/phpunit --testdox --colors=never"
    exts: list[str] = field(default_factory=lambda: [".php"])
    _fqcn_to_file_cache: dict[str, str] = field(default=None, init=False, repr=False)

    def _build_fqcn_to_file_map(self) -> dict[str, str]:
        """Build a mapping from fully-qualified class names to file paths.

        Scans PHP test files for namespace + class declarations.
        """
        dest, cloned = self.clone()
        fqcn_to_file: dict[str, str] = {}

        namespace_re = re.compile(r"^\s*namespace\s+([\w\\]+)\s*;")
        class_re = re.compile(r"^\s*(?:abstract\s+|final\s+)?class\s+(\w+)")

        for dirpath, _, filenames in os.walk(dest):
            if "vendor" in dirpath.split(os.sep):
                continue
            for fname in filenames:
                if not fname.endswith(".php"):
                    continue
                parts = dirpath.split(os.sep)
                in_test_dir = any(
                    p in ("test", "tests", "Test", "Tests") for p in parts
                )
                is_test_named = fname.endswith("Test.php") or fname.startswith("test")
                if not in_test_dir and not is_test_named:
                    continue

                full_path = os.path.join(dirpath, fname)
                relative_path = os.path.relpath(full_path, dest)

                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except (OSError, UnicodeDecodeError):
                    continue

                namespace = ""
                for line in content.split("\n"):
                    ns_match = namespace_re.match(line)
                    if ns_match:
                        namespace = ns_match.group(1)
                        break

                for line in content.split("\n"):
                    cls_match = class_re.match(line)
                    if cls_match:
                        fqcn = cls_match.group(1)
                        if namespace:
                            fqcn = f"{namespace}\\{fqcn}"
                        fqcn_to_file[fqcn] = relative_path
                        break

        if cloned:
            shutil.rmtree(dest)
        return fqcn_to_file

    def get_test_files(self, instance: dict) -> tuple[list[str], list[str]]:
        assert FAIL_TO_PASS in instance and PASS_TO_PASS in instance, (
            f"Instance {instance[KEY_INSTANCE_ID]} missing required keys {FAIL_TO_PASS} or {PASS_TO_PASS}"
        )

        if self._fqcn_to_file_cache is None:
            with self._lock:
                if self._fqcn_to_file_cache is None:
                    self._fqcn_to_file_cache = self._build_fqcn_to_file_map()

        def _resolve_files(test_names: list[str]) -> set[str]:
            files: set[str] = set()
            for test_name in test_names:
                # test_name is "FQCN::Testdox name" — extract the FQCN
                fqcn = test_name.split("::")[0] if "::" in test_name else test_name
                if fqcn in self._fqcn_to_file_cache:
                    files.add(self._fqcn_to_file_cache[fqcn])
            return files

        f2p_files = _resolve_files(instance[FAIL_TO_PASS])
        p2p_files = _resolve_files(instance[PASS_TO_PASS])
        return list(f2p_files), list(p2p_files)


@dataclass
class Dbalacb68b38(PhpProfile):
    owner: str = "doctrine"
    repo: str = "dbal"
    commit: str = "acb68b388b2577bb211bb26dc22d20a8ad93d97d"

    @property
    def dockerfile(self):
        return f"""FROM php:8.3
RUN apt-get update && \
    apt-get install -y wget git build-essential unzip libgd-dev libzip-dev libgmp-dev libftp-dev libcurl4-openssl-dev libpq-dev libsqlite3-dev && \
    docker-php-ext-install pdo pdo_mysql pdo_pgsql pdo_sqlite mysqli gd zip gmp ftp curl pcntl && \
    apt-get -y autoclean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sS https://getcomposer.org/installer | php -- --2.2 --install-dir=/usr/local/bin --filename=composer

RUN git clone https://github.com/{self.mirror_name} /{ENV_NAME}
WORKDIR /{ENV_NAME}
RUN composer update
RUN composer install
"""

    def log_parser(self, log: str) -> dict[str, str]:
        return parse_log_phpunit_testdox(log)


def parse_log_phpunit_testdox(log: str) -> dict[str, str]:
    """Parse PHPUnit --testdox output format.

    Captures class headers like:
        Connection (Doctrine\\DBAL\\Tests\\ConnectionTest)
    and qualifies test names as 'FQCN::Testdox name'.
    """
    test_status_map = {}
    class_header = re.compile(r"^.+\((.+)\)\s*$")
    passed_pattern = re.compile(r"^\s*✔\s*(.+)$")
    failed_pattern = re.compile(r"^\s*✘\s*(.+)$")
    skipped_pattern = re.compile(r"^\s*↩\s*(.+)$")
    current_class = None
    for line in log.split("\n"):
        cm = class_header.match(line)
        if cm:
            current_class = cm.group(1).strip()
            continue
        for pattern, status in (
            (passed_pattern, TestStatus.PASSED.value),
            (failed_pattern, TestStatus.FAILED.value),
            (skipped_pattern, TestStatus.SKIPPED.value),
        ):
            match = pattern.match(line)
            if match:
                test_name = match.group(1).strip()
                if current_class:
                    test_name = f"{current_class}::{test_name}"
                test_status_map[test_name] = status
                break
    return test_status_map


@dataclass
class Guzzlefb92d95f(PhpProfile):
    owner: str = "guzzle"
    repo: str = "guzzle"
    commit: str = "fb92d95f80a9da51bf8f2a5b26d8e8ea3b6d99ed"

    @property
    def dockerfile(self):
        return f"""FROM php:8.3
RUN apt-get update && \
    apt-get install -y git unzip libcurl4-openssl-dev libzip-dev nodejs npm && \
    docker-php-ext-install curl zip && \
    apt-get -y autoclean && \
    rm -rf /var/lib/apt/lists/*
RUN curl -sS https://getcomposer.org/installer | php -- --install-dir=/usr/local/bin --filename=composer
RUN git clone https://github.com/{self.mirror_name} /{ENV_NAME}
WORKDIR /{ENV_NAME}
RUN composer install
"""

    def log_parser(self, log: str) -> dict[str, str]:
        return parse_log_phpunit_testdox(log)


@dataclass
class Monolog6db20ca0(PhpProfile):
    owner: str = "Seldaek"
    repo: str = "monolog"
    commit: str = "6db20ca029219dd8de378cea8e32ee149399ef1b"

    @property
    def dockerfile(self):
        return f"""FROM php:8.3
RUN apt-get update && \
    apt-get install -y git unzip libcurl4-openssl-dev && \
    docker-php-ext-install curl sockets && \
    apt-get -y autoclean && \
    rm -rf /var/lib/apt/lists/*
RUN curl -sS https://getcomposer.org/installer | php -- --install-dir=/usr/local/bin --filename=composer
RUN git clone https://github.com/{self.mirror_name} /{ENV_NAME}
WORKDIR /{ENV_NAME}
RUN composer install --ignore-platform-req=ext-mongodb
"""

    def log_parser(self, log: str) -> dict[str, str]:
        return parse_log_phpunit_testdox(log)


# Register all PHP profiles with the global registry
for name, obj in list(globals().items()):
    if (
        isinstance(obj, type)
        and issubclass(obj, PhpProfile)
        and obj.__name__ != "PhpProfile"
    ):
        registry.register_profile(obj)
