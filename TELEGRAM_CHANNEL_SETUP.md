# Telegram Channel Setup Guide for AENEAS

## Quick Setup

### 1. Edit the .env File

Open your `.env` file and add channels to line 35:

```bash
# Single channel
TELEGRAM_CHANNELS=cryptowhale

# Multiple channels (comma-separated, no spaces)
TELEGRAM_CHANNELS=cryptowhale,binancesignals,tradingpro,vipsignals

# Mix of public usernames and private IDs
TELEGRAM_CHANNELS=cryptowhale,-1001234567890,tradingpro,-1009876543210

# With @ prefix (also valid)
TELEGRAM_CHANNELS=@cryptowhale,@binancesignals,@tradingpro
```

## Detailed Instructions

### Finding Channel Identifiers

#### Method 1: Public Channels (Easiest)
1. Open the channel in Telegram
2. Click channel name at top
3. Look for "Username: @channelname"
4. Use either `channelname` or `@channelname`

#### Method 2: Using Telegram Web
1. Open https://web.telegram.org
2. Navigate to the channel
3. Look at the URL: `https://web.telegram.org/z/#-100XXXXXXXXX`
4. Copy the number including the minus sign

#### Method 3: Using a Bot
1. Add @userinfobot to Telegram
2. Forward any message from the target channel to the bot
3. Bot will reply with channel ID

#### Method 4: Using Telegram Desktop
1. Right-click on channel
2. Select "Copy Link"
3. If it shows `t.me/channelname` - use `channelname`
4. If private, you'll need to use one of the other methods

### Channel Types and Formats

| Channel Type | Example | Format in .env |
|-------------|---------|----------------|
| Public Channel | t.me/cryptowhale | `cryptowhale` or `@cryptowhale` |
| Private Channel | No public link | `-1001234567890` (use ID) |
| Supergroup | Group upgraded to supergroup | `-1001234567890` (use ID) |
| Bot Channel | Channel managed by bot | `@botchannelname` or ID |

### Configuration Examples

#### Example 1: Crypto Trading Channels
```env
TELEGRAM_CHANNELS=@binancekillers,@cryptovipsignals,@wolfxsignals,@margincalls
```

#### Example 2: Mixed Public and Private
```env
TELEGRAM_CHANNELS=publicchannel1,-1001234567890,publicchannel2,-1009876543210
```

#### Example 3: Test Configuration
```env
# Start with one channel for testing
TELEGRAM_CHANNELS=cryptowhale

# Then expand to multiple
TELEGRAM_CHANNELS=cryptowhale,binancesignals,kucoinpump
```

### Permissions Required

The Telegram account (phone number in .env) must:
1. **Be a member** of private channels/groups
2. **Have read access** to the channels
3. **Not be restricted** or banned from the channels

### Adding Channels Dynamically

You can also add channels via API after the system is running:

```python
# Using the API endpoint
POST /api/v1/collector/channels
{
    "channel_id": "@newchannel",
    "active": true
}
```

Or via database:
```sql
INSERT INTO telegram_channels (username, channel_id, active) 
VALUES ('@cryptowhale', '-1001234567890', true);
```

### Verification Steps

After adding channels:

1. **Test Connection**:
```bash
python3 -c "from src.data_ingestion.telegram_collector import TelegramCollector; import asyncio; asyncio.run(TelegramCollector().test_channels())"
```

2. **Check Logs**:
```bash
docker logs crypto_signals_app | grep "Monitoring channel"
```

3. **Verify in Database**:
```sql
SELECT * FROM telegram_channels WHERE active = true;
```

### Troubleshooting

#### Channel Not Found
- Verify the username is correct (case-sensitive)
- Ensure no typos or extra spaces
- Try with and without @ prefix

#### Access Denied
- Make sure your account is a member
- Check if channel requires admin approval
- Verify phone number in TELEGRAM_PHONE_NUMBER

#### No Messages Received
- Channel might be inactive
- Check TELEGRAM_API_ID and TELEGRAM_API_HASH
- Ensure session file isn't corrupted

### Best Practices

1. **Start Small**: Begin with 1-2 channels for testing
2. **Quality Over Quantity**: Choose reputable signal channels
3. **Monitor Performance**: Track accuracy per channel
4. **Regular Updates**: Remove inactive/poor performing channels
5. **Backup Channel List**: Keep a backup of good channel IDs

### Security Notes

- **Never share** your TELEGRAM_API_ID and TELEGRAM_API_HASH
- **Use a dedicated account** for the bot, not your personal account
- **Monitor usage** to avoid rate limiting
- **Rotate sessions** periodically for security

## Quick Reference

```env
# Your .env file (line ~35)
TELEGRAM_CHANNELS=channel1,channel2,channel3,-1001234567890,@channel5

# Common signal channels (examples)
TELEGRAM_CHANNELS=@cryptowhale,@binancekillers,@marginsignals,@kucoinpump,@bybitscalpers

# Test configuration
TELEGRAM_CHANNELS=@your_test_channel
```

## Next Steps

1. Add your channels to `.env`
2. Restart the application:
   ```bash
   docker-compose restart
   # or
   python3 src/main.py
   ```
3. Monitor logs for successful connection
4. Check the database for incoming messages
5. Use the API to manage channels dynamically
