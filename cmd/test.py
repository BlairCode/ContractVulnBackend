import solcx

# 指定要下载的版本
versions = {'0.5.11', '0.6.12', '0.4.12', '0.4.17', '0.4.25', '0.4.19', '0.4.24', '0.6.11', '0.4.22', '0.4.11', '0.4.99', '0.4.23', '0.4.16', '0.5.15', '0.4.13', '0.4.21', '0.5.17', '0.4.20', '0.4.26', '0.4.18', '0.4.15', '0.5.16', '0.4.14', '0.5.14', '0.5.13', '0.5.12'}

# 获取已安装的版本
installed_versions = solcx.get_installed_solc_versions()

# 批量下载
for version in versions:
    # 检查是否已安装
    if version in installed_versions:
        print(f"Version {version} already installed. Skipping...")
        continue
    
    try:
        print(f"Downloading Solidity version {version}...")
        solcx.install_solc(version, show_progress=True)
        print(f"Successfully installed version {version}")
    except Exception as e:
        print(f"Failed to install version {version}: {e}")

print("Solidity compiler installation complete.")

# print(solcx.get_installable_solc_versions())