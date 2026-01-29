//! Generate a new Polygon wallet for Polymarket

use ethers::core::k256::ecdsa::SigningKey;
use ethers::core::rand::thread_rng;
use ethers::signers::{LocalWallet, Signer};

fn main() {
    // Generate random signing key using ethers' rand
    let signing_key = SigningKey::random(&mut thread_rng());
    let wallet = LocalWallet::from(signing_key);
    
    // Get private key bytes
    let pk_bytes = wallet.signer().to_bytes();
    let pk_hex: String = pk_bytes.iter().map(|b| format!("{:02x}", b)).collect();
    
    println!("🔐 New Polymarket Wallet Generated\n");
    println!("Address: {:?}", wallet.address());
    println!("Private Key: {}", pk_hex);
    println!("\n⚠️  IMPORTANT: Save the private key securely!");
    println!("⚠️  Never share it with anyone!");
    println!("\n📝 Add to config.toml:");
    println!("private_key = \"{}\"", pk_hex);
}
