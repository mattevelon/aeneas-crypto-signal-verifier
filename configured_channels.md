# Configured Telegram Channels

## Channels Added to .env (2025-09-16)

Successfully added 17 channels to the TELEGRAM_CHANNELS configuration:

### Active Channels
1. **Coin_Post** - https://t.me/Coin_Post
2. **ne_investor** - https://t.me/ne_investor  
3. **cryptoc** - https://t.me/cryptoc
4. **wellfedhamster** - https://t.me/wellfedhamster
5. **ru_holder** - https://t.me/ru_holder
6. **wolfoftrading** - https://t.me/wolfoftrading
7. **cryptoinnercircle** - https://t.me/cryptoinnercircle
8. **binancekillers** - https://t.me/binancekillers
9. **crypto_pumps_p** - https://t.me/crypto_pumps_p
10. **FedRussianInsiders** - https://t.me/FedRussianInsiders
11. **CryptoRankNews** - https://t.me/CryptoRankNews
12. **BeInCryptoCommunity** - https://t.me/BeInCryptoCommunity
13. **RAVENSignalspro_io** - https://t.me/RAVENSignalspro_io
14. **trade** - https://t.me/trade
15. **sirberezanews** - https://t.me/sirberezanews
16. **darkside_trades** - https://t.me/darkside_trades
17. **Private Channel** - ID: -1002663923876 (from invite link)

### Special Cases

#### Private Channel (Added via ID)
- **Channel ID: -1002663923876**
  - This was the invite link channel: https://t.me/+HW-VLBY7DLkwOWUy
  - Added using the channel ID after joining
  - Now actively monitored

#### Message Link
- **https://t.me/BeInCryptoCommunity/493394**
  - Added as "BeInCryptoCommunity" (channel name only)
  - The /493394 refers to a specific message, not needed for monitoring

## How to Update

To add the invite link channel after joining:

1. Join the channel using the invite link
2. Get the channel ID using one of these methods:
   - Forward a message to @userinfobot
   - Check URL in Telegram Web
3. Add to .env file:
   ```bash
   # Current channels + new ID
   TELEGRAM_CHANNELS=Coin_Post,ne_investor,...,darkside_trades,-100XXXXXXXXX
   ```

## Verification

To verify channels are working:

```bash
# Check configuration
grep TELEGRAM_CHANNELS .env

# Test connection (after starting the app)
python3 -c "
from src.config.settings import settings
channels = settings.telegram_channels.split(',')
print(f'Configured {len(channels)} channels:')
for ch in channels:
    print(f'  - {ch}')
"
```

## Notes
- All channels are public except the invite link
- Channels are monitored for trading signals
- The system will automatically filter and analyze messages
- Channel performance is tracked in the database
