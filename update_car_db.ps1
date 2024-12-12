# Variables
$LocalFile = "C:\Users\pgrts\Desktop\python\car_proj\data\car_db.db" # Path to your local file
$VMName = "app-vm" # Name of your Google Cloud VM
$Zone = "us-central1-a" # Zone of your VM
$RemotePath = "/mnt/disks/gce-containers-mounts/gce-persistent-disks/car-db-disk/car_db.db" # Destination on the VM
$User = "mitchag191" # SSH user (likely your Google Cloud username)
$VMIP = "34.136.95.169" # Optional: Replace with static IP if needed
$PrivateKeyPath = "C:\Users\pgrts\Desktop\gkey\google_compute_engine"


# Backup the old file
ssh -i $PrivateKeyPath ${User}@${VMIP} "cp ${RemotePath} ${RemotePath}.bak"

# File Transfer using scp
Write-Host "Transferring file to VM..."
scp -i $PrivateKeyPath $LocalFile "${User}@${VMIP}:${RemotePath}"

# Verify the Transfer
Write-Host "Verifying the file on the remote VM..."
ssh -i $PrivateKeyPath ${User}@${VMIP} "ls -lh ${RemotePath}"

Write-Host "File transfer and verification complete."