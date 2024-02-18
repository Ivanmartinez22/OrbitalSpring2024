# Use this script to test whether the current package requirements are satisfied

import pkg_resources

# Tests parameters against current environment packages and versions
def checkPackageVersion(package, version):
    try:
        currentVersion = pkg_resources.get_distribution(package).version

        # Raise AssertionError if package version is too low
        if pkg_resources.parse_version(currentVersion) < pkg_resources.parse_version(version):
            raise AssertionError(f"\n{package} is installed. Current Version: {currentVersion} || Version Required: {version}\n")
    
    # Raise AssertionError if package not installed
    except pkg_resources.DistributionNotFound:
        raise AssertionError(f"\n{package} is not installed.\n")
    
def testEnvironment():
    # Reads requirements.txt file
    with open('requirements.txt', 'r') as f:
        lines=f.readlines()
    
    # Parses each line to identify necessary package and version
    for line in lines:
        if line != "\n":
            result = line.strip().split('==', maxsplit=1)
            package, version = result[0], result[1]
            checkPackageVersion(package, version)
    
    print("All requirements satisfied!")

#testEnvironment()