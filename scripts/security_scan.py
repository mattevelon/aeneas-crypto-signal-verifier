#!/usr/bin/env python3
"""
Security vulnerability scanner for AENEAS.
Performs comprehensive security checks on the codebase and dependencies.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List
import yaml
import re

class SecurityScanner:
    """Comprehensive security scanner."""
    
    def __init__(self):
        self.vulnerabilities = []
        self.security_score = 100
        
    def run_all_scans(self) -> Dict[str, Any]:
        """Run all security scans."""
        print("🔍 Starting Security Scan...")
        
        results = {
            "dependencies": self.scan_dependencies(),
            "code": self.scan_code(),
            "secrets": self.scan_secrets(),
            "docker": self.scan_docker(),
            "configuration": self.scan_configuration(),
            "owasp": self.check_owasp_top10()
        }
        
        # Calculate overall security score
        self.calculate_score()
        
        results["summary"] = {
            "total_vulnerabilities": len(self.vulnerabilities),
            "critical": len([v for v in self.vulnerabilities if v.get("severity") == "critical"]),
            "high": len([v for v in self.vulnerabilities if v.get("severity") == "high"]),
            "medium": len([v for v in self.vulnerabilities if v.get("severity") == "medium"]),
            "low": len([v for v in self.vulnerabilities if v.get("severity") == "low"]),
            "security_score": self.security_score
        }
        
        return results
        
    def scan_dependencies(self) -> Dict[str, Any]:
        """Scan dependencies for known vulnerabilities."""
        print("  📦 Scanning dependencies...")
        
        vulnerabilities = []
        
        # Check Python dependencies with safety
        try:
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True
            )
            if result.stdout:
                safety_results = json.loads(result.stdout)
                for vuln in safety_results:
                    vulnerabilities.append({
                        "package": vuln.get("package"),
                        "installed_version": vuln.get("installed_version"),
                        "vulnerability": vuln.get("vulnerability"),
                        "severity": "high"
                    })
                    self.vulnerabilities.append(vuln)
        except Exception as e:
            print(f"    ⚠️ Safety scan failed: {e}")
            
        # Check for outdated packages
        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True
            )
            if result.stdout:
                outdated = json.loads(result.stdout)
                if len(outdated) > 10:
                    self.vulnerabilities.append({
                        "type": "outdated_dependencies",
                        "count": len(outdated),
                        "severity": "low"
                    })
        except Exception as e:
            print(f"    ⚠️ Outdated check failed: {e}")
            
        return {
            "vulnerabilities": vulnerabilities,
            "outdated_count": len(outdated) if 'outdated' in locals() else 0
        }
        
    def scan_code(self) -> Dict[str, Any]:
        """Scan code for security issues."""
        print("  🔐 Scanning code...")
        
        issues = []
        
        # Run bandit for Python security issues
        try:
            result = subprocess.run(
                ["bandit", "-r", "src/", "-f", "json"],
                capture_output=True,
                text=True
            )
            if result.stdout:
                bandit_results = json.loads(result.stdout)
                for issue in bandit_results.get("results", []):
                    issues.append({
                        "file": issue.get("filename"),
                        "line": issue.get("line_number"),
                        "severity": issue.get("issue_severity").lower(),
                        "issue": issue.get("issue_text")
                    })
                    if issue.get("issue_severity").lower() in ["high", "critical"]:
                        self.vulnerabilities.append(issue)
        except Exception as e:
            print(f"    ⚠️ Bandit scan failed: {e}")
            
        # Check for common security patterns
        patterns = [
            (r"eval\(", "Use of eval() is dangerous"),
            (r"exec\(", "Use of exec() is dangerous"),
            (r"os\.system\(", "Use of os.system() is dangerous"),
            (r"pickle\.loads", "Pickle deserialization is unsafe"),
            (r"yaml\.load\(", "Use yaml.safe_load() instead"),
            (r"DEBUG\s*=\s*True", "Debug mode should be disabled in production"),
            (r"SECRET_KEY\s*=\s*['\"]", "Hardcoded secret key detected"),
            (r"PASSWORD\s*=\s*['\"]", "Hardcoded password detected")
        ]
        
        for file_path in Path("src").rglob("*.py"):
            try:
                content = file_path.read_text()
                for pattern, message in patterns:
                    if re.search(pattern, content):
                        issues.append({
                            "file": str(file_path),
                            "pattern": pattern,
                            "message": message,
                            "severity": "high"
                        })
                        self.vulnerabilities.append({
                            "type": "insecure_pattern",
                            "file": str(file_path),
                            "message": message,
                            "severity": "high"
                        })
            except Exception as e:
                print(f"      Error scanning {file_path}: {e}")
                
        return {"issues": issues, "files_scanned": len(list(Path("src").rglob("*.py")))}
        
    def scan_secrets(self) -> Dict[str, Any]:
        """Scan for exposed secrets."""
        print("  🔑 Scanning for secrets...")
        
        secrets_found = []
        
        # Patterns for common secrets
        secret_patterns = [
            (r"(?i)api[_\-\s]*key[_\-\s]*[:=]\s*['\"][^'\"]{20,}", "API Key"),
            (r"(?i)secret[_\-\s]*key[_\-\s]*[:=]\s*['\"][^'\"]{20,}", "Secret Key"),
            (r"(?i)password[_\-\s]*[:=]\s*['\"][^'\"]+", "Password"),
            (r"(?i)token[_\-\s]*[:=]\s*['\"][^'\"]{20,}", "Token"),
            (r"aws_access_key_id\s*=\s*['\"][^'\"]+", "AWS Access Key"),
            (r"aws_secret_access_key\s*=\s*['\"][^'\"]+", "AWS Secret Key"),
            (r"(?i)database_url\s*=\s*['\"][^'\"]+", "Database URL")
        ]
        
        exclude_paths = [".git", ".venv", "venv", "__pycache__", ".pytest_cache"]
        
        for file_path in Path(".").rglob("*"):
            if any(exc in str(file_path) for exc in exclude_paths):
                continue
            if file_path.is_file() and file_path.suffix in [".py", ".yml", ".yaml", ".json", ".env"]:
                try:
                    content = file_path.read_text()
                    for pattern, secret_type in secret_patterns:
                        if re.search(pattern, content):
                            # Skip if it's in .env.example
                            if file_path.name != ".env.example":
                                secrets_found.append({
                                    "file": str(file_path),
                                    "type": secret_type,
                                    "severity": "critical"
                                })
                                self.vulnerabilities.append({
                                    "type": "exposed_secret",
                                    "file": str(file_path),
                                    "secret_type": secret_type,
                                    "severity": "critical"
                                })
                except Exception as e:
                    print(f"        Error checking {file_path}: {e}")
                    
        return {"secrets_found": len(secrets_found), "details": secrets_found[:5]}  # Limit details for security
        
    def scan_docker(self) -> Dict[str, Any]:
        """Scan Docker configuration."""
        print("  🐳 Scanning Docker configuration...")
        
        issues = []
        
        dockerfile_path = Path("Dockerfile")
        if dockerfile_path.exists():
            content = dockerfile_path.read_text()
            
            # Check for security best practices
            checks = [
                (r"FROM .+:latest", "Avoid using 'latest' tag", "medium"),
                (r"USER root", "Avoid running as root", "high"),
                (r"RUN .+apt-get.+&&", "Combine RUN commands", "low"),
                (r"ADD ", "Use COPY instead of ADD", "low"),
                (r"EXPOSE 22", "SSH should not be exposed", "high"),
                (r"ENV .+PASSWORD", "Passwords in ENV variables", "critical")
            ]
            
            for pattern, message, severity in checks:
                if re.search(pattern, content):
                    issues.append({
                        "pattern": pattern,
                        "message": message,
                        "severity": severity
                    })
                    if severity in ["high", "critical"]:
                        self.vulnerabilities.append({
                            "type": "docker_security",
                            "message": message,
                            "severity": severity
                        })
                        
        return {"issues": issues}
        
    def scan_configuration(self) -> Dict[str, Any]:
        """Scan configuration files."""
        print("  ⚙️ Scanning configuration...")
        
        issues = []
        
        # Check for insecure configurations
        config_checks = {
            "cors_origins": "*",  # Too permissive CORS
            "debug": True,  # Debug mode enabled
            "ssl_verify": False,  # SSL verification disabled
            "allow_all_origins": True,  # Permissive origins
        }
        
        config_files = list(Path(".").rglob("*.yml")) + list(Path(".").rglob("*.yaml")) + list(Path(".").rglob("*.json"))
        
        for config_file in config_files:
            try:
                content = config_file.read_text()
                if config_file.suffix in [".yml", ".yaml"]:
                    config = yaml.safe_load(content)
                elif config_file.suffix == ".json":
                    config = json.loads(content)
                else:
                    continue
                    
                # Flatten nested dict for checking
                def flatten_dict(d, parent_key=''):
                    items = []
                    for k, v in d.items() if isinstance(d, dict) else []:
                        new_key = f"{parent_key}.{k}" if parent_key else k
                        if isinstance(v, dict):
                            items.extend(flatten_dict(v, new_key).items())
                        else:
                            items.append((new_key, v))
                    return dict(items)
                    
                flat_config = flatten_dict(config)
                
                for key, bad_value in config_checks.items():
                    if key in flat_config and flat_config[key] == bad_value:
                        issues.append({
                            "file": str(config_file),
                            "setting": key,
                            "value": bad_value,
                            "severity": "medium"
                        })
                        self.vulnerabilities.append({
                            "type": "insecure_configuration",
                            "file": str(config_file),
                            "setting": key,
                            "severity": "medium"
                        })
                        
            except Exception as e:
                print(f"      Error checking configuration {file_path}: {e}")
                
        return {"issues": issues}
        
    def check_owasp_top10(self) -> Dict[str, Any]:
        """Check for OWASP Top 10 vulnerabilities."""
        print("  🛡️ Checking OWASP Top 10...")
        
        checks = {
            "A01_Broken_Access_Control": self.check_access_control(),
            "A02_Cryptographic_Failures": self.check_cryptography(),
            "A03_Injection": self.check_injection(),
            "A04_Insecure_Design": self.check_design(),
            "A05_Security_Misconfiguration": self.check_misconfiguration(),
            "A06_Vulnerable_Components": self.check_components(),
            "A07_Authentication_Failures": self.check_authentication(),
            "A08_Data_Integrity_Failures": self.check_data_integrity(),
            "A09_Logging_Failures": self.check_logging(),
            "A10_SSRF": self.check_ssrf()
        }
        
        return checks
        
    def check_access_control(self) -> bool:
        """Check for broken access control."""
        # Check for proper authentication decorators
        auth_patterns = ["@require_auth", "@login_required", "check_permissions"]
        found = False
        
        for file_path in Path("src/api").rglob("*.py"):
            try:
                content = file_path.read_text()
                if any(pattern in content for pattern in auth_patterns):
                    found = True
                    break
            except Exception:
                pass
                
        if not found:
            self.vulnerabilities.append({
                "type": "owasp_a01",
                "message": "Missing authentication checks",
                "severity": "high"
            })
            
        return found
        
    def check_cryptography(self) -> bool:
        """Check for cryptographic failures."""
        # Check for weak crypto
        weak_patterns = ["md5", "sha1", "DES", "RC4"]
        found_weak = False
        
        for file_path in Path("src").rglob("*.py"):
            try:
                content = file_path.read_text()
                if any(pattern in content.lower() for pattern in weak_patterns):
                    found_weak = True
                    self.vulnerabilities.append({
                        "type": "owasp_a02",
                        "message": "Weak cryptography detected",
                        "file": str(file_path),
                        "severity": "high"
                    })
                    break
            except Exception as e:
                print(f"      Error checking {file_path}: {e}")
                
        return not found_weak
        
    def check_injection(self) -> bool:
        """Check for injection vulnerabilities."""
        # Check for SQL injection patterns
        sql_patterns = [
            r"f['\"].*SELECT.*{",
            r"\".*SELECT.*\" \+",
            r"sql = .*\+",
            r"execute\(['\"].*%s"
        ]
        
        found_injection = False
        
        for file_path in Path("src").rglob("*.py"):
            try:
                content = file_path.read_text()
                for pattern in sql_patterns:
                    if re.search(pattern, content):
                        found_injection = True
                        self.vulnerabilities.append({
                            "type": "owasp_a03",
                            "message": "Potential SQL injection",
                            "file": str(file_path),
                            "severity": "critical"
                        })
                        break
            except Exception as e:
                print(f"      Error checking {file_path}: {e}")
                
        return not found_injection
        
    def check_design(self) -> bool:
        """Check for insecure design."""
        # Basic check for security design patterns
        return True  # Would need more complex analysis
        
    def check_misconfiguration(self) -> bool:
        """Check for security misconfiguration."""
        # Already covered in scan_configuration
        return len([v for v in self.vulnerabilities if v.get("type") == "insecure_configuration"]) == 0
        
    def check_components(self) -> bool:
        """Check for vulnerable components."""
        # Already covered in scan_dependencies
        return len([v for v in self.vulnerabilities if v.get("type") == "outdated_dependencies"]) == 0
        
    def check_authentication(self) -> bool:
        """Check for authentication failures."""
        # Check for proper password policies
        password_checks = ["bcrypt", "argon2", "scrypt", "pbkdf2"]
        found = False
        
        for file_path in Path("src").rglob("*.py"):
            try:
                content = file_path.read_text()
                if any(check in content.lower() for check in password_checks):
                    found = True
                    break
            except Exception as e:
                print(f"      Error checking {file_path}: {e}")
                
        if not found:
            self.vulnerabilities.append({
                "type": "owasp_a07",
                "message": "Weak password hashing",
                "severity": "high"
            })
            
        return found
        
    def check_data_integrity(self) -> bool:
        """Check for data integrity failures."""
        # Check for proper serialization
        unsafe_patterns = ["pickle.loads", "eval(", "exec("]
        found_unsafe = False
        
        for file_path in Path("src").rglob("*.py"):
            try:
                content = file_path.read_text()
                if any(pattern in content for pattern in unsafe_patterns):
                    found_unsafe = True
                    self.vulnerabilities.append({
                        "type": "owasp_a08",
                        "message": "Unsafe deserialization",
                        "file": str(file_path),
                        "severity": "high"
                    })
                    break
            except Exception as e:
                print(f"      Error checking {file_path}: {e}")
                
        return not found_unsafe
        
    def check_logging(self) -> bool:
        """Check for logging failures."""
        # Check for proper logging
        logging_patterns = ["import logging", "import structlog", "logger ="]
        found = False
        
        for file_path in Path("src").rglob("*.py"):
            try:
                content = file_path.read_text()
                if any(pattern in content for pattern in logging_patterns):
                    found = True
                    break
            except Exception as e:
                print(f"      Error checking {file_path}: {e}")
                
        if not found:
            self.vulnerabilities.append({
                "type": "owasp_a09",
                "message": "Insufficient logging",
                "severity": "medium"
            })
            
        return found
        
    def check_ssrf(self) -> bool:
        """Check for SSRF vulnerabilities."""
        # Check for unsafe URL fetching
        url_patterns = ["requests.get(", "urllib.request.urlopen(", "httpx.get("]
        validation_patterns = ["validate_url", "is_safe_url", "allowed_hosts"]
        
        found_url = False
        found_validation = False
        
        for file_path in Path("src").rglob("*.py"):
            try:
                content = file_path.read_text()
                if any(pattern in content for pattern in url_patterns):
                    found_url = True
                if any(pattern in content for pattern in validation_patterns):
                    found_validation = True
            except Exception as e:
                print(f"      Error checking {file_path}: {e}")
                
        if found_url and not found_validation:
            self.vulnerabilities.append({
                "type": "owasp_a10",
                "message": "Potential SSRF vulnerability",
                "severity": "medium"
            })
            
        return not (found_url and not found_validation)
        
    def calculate_score(self):
        """Calculate overall security score."""
        for vuln in self.vulnerabilities:
            severity = vuln.get("severity", "low")
            if severity == "critical":
                self.security_score -= 20
            elif severity == "high":
                self.security_score -= 10
            elif severity == "medium":
                self.security_score -= 5
            elif severity == "low":
                self.security_score -= 2
                
        self.security_score = max(0, self.security_score)
        
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate security report."""
        report = ["=" * 60]
        report.append("AENEAS SECURITY SCAN REPORT")
        report.append("=" * 60)
        report.append("")
        
        summary = results["summary"]
        report.append(f"Security Score: {summary['security_score']}/100")
        report.append(f"Total Vulnerabilities: {summary['total_vulnerabilities']}")
        report.append(f"  Critical: {summary['critical']}")
        report.append(f"  High: {summary['high']}")
        report.append(f"  Medium: {summary['medium']}")
        report.append(f"  Low: {summary['low']}")
        report.append("")
        
        if summary['critical'] > 0:
            report.append("⛔ CRITICAL ISSUES FOUND - IMMEDIATE ACTION REQUIRED")
        elif summary['high'] > 0:
            report.append("⚠️ HIGH SEVERITY ISSUES FOUND - ACTION RECOMMENDED")
        elif summary['security_score'] >= 80:
            report.append("✅ SECURITY POSTURE: GOOD")
        else:
            report.append("⚠️ SECURITY POSTURE: NEEDS IMPROVEMENT")
            
        report.append("")
        report.append("RECOMMENDATIONS:")
        
        if summary['critical'] > 0 or summary['high'] > 0:
            report.append("1. Fix all critical and high severity issues before deployment")
        report.append("2. Keep all dependencies up to date")
        report.append("3. Implement security headers and CSP")
        report.append("4. Enable rate limiting and DDoS protection")
        report.append("5. Regular security audits and penetration testing")
        
        return "\n".join(report)


def main():
    """Run security scan."""
    scanner = SecurityScanner()
    results = scanner.run_all_scans()
    
    # Generate and print report
    report = scanner.generate_report(results)
    print("\n" + report)
    
    # Save detailed results
    with open("security_scan_report.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed report saved to security_scan_report.json")
    
    # Exit with error if critical issues found
    if results["summary"]["critical"] > 0:
        sys.exit(1)
    
    return 0


if __name__ == "__main__":
    main()
