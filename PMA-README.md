# Waves: Personalized Meditation App

A personalized meditation app that delivers tailored meditation sessions based on user input. The app uses OpenAI's GPT models to generate meditation scripts and text-to-speech to create audio experiences, with a clean, calming interface designed for iOS users.

## Features

- User authentication (email/password and Apple Sign-in with secure password reset)
- Text and speech input for personalized meditation requests
- LLM-generated meditation scripts based on user input
- Text-to-speech conversion of scripts to audio
- Customizable meditation lengths (5-30+ minutes)
- Storage and retrieval of past meditation sessions
- Time-based UI themes that change throughout the day (sunrise, surface, standard, moonlight)
- Animated wave overlay for enhanced visual experience
- Personalized meditation suggestions based on user history
- Credit-based monetization system (1 credit = 5 minutes of meditation)
- User feedback system with thumbs up/down for meditation quality
- FAQ section with helpful information
- Contact Us feature for user support
- Personalized welcome messages based on time of day
- Interactive prompt carousel with category-based suggestions and seamless animation

## Technical Stack

- **Platform**: iOS/Swift for frontend with SwiftUI (iPhone only, portrait orientation only)
- **Architecture**: MVVM (Model-View-ViewModel) pattern
- **Authentication**: Supabase with email/password and Apple Sign-in integration (using Keychain for secure session storage)
- **Database**: Supabase PostgreSQL for user data, meditation storage, and purchase history
- **Storage**: Supabase Storage for audio files and user assets
- **AI Integration**:
  - OpenAI GPT-4o for speech-to-text conversion
  - OpenAI o3-mini for meditation script generation and personalization algorithm
  - OpenAI Text-to-Speech API for audio output with voice customization
- **Monetization**: StoreKit 2 for in-app purchases and subscriptions with transaction verification

## Getting Started

1. Clone the repository
2. Install dependencies using Swift Package Manager
3. Create a `Config.swift` file in the Helpers directory with your API keys:
   ```swift
   struct Config {
     static let shared = Config()
     let supabaseURL: String = "YOUR_SUPABASE_URL"
     let supabaseAnonKey: String = "YOUR_SUPABASE_ANON_KEY"
     let openAIAPIKey: String = "YOUR_OPENAI_API_KEY"
     let supabaseServiceRoleKey: String = "YOUR_SERVICE_ROLE_KEY" // For edge functions
   }
   ```
4. Set up database tables in Supabase dashboard (run in order):
   - `Database/create_app_store_tables.sql` - Monetization tables
   - `Database/add_user_credits.sql` - Credit system
   - `Database/setup_user_credits_rls.sql` - Row-level security
   - `Database/create_user_credits_trigger.sql` - Auto-credit on signup
5. Deploy Edge Functions:
   ```bash
   supabase functions deploy register-user
   supabase functions deploy app-store-notifications
   ```
6. Set up your Apple Developer account for StoreKit testing
7. Build and run the app in Xcode
8. Note that this app is designed for iPhone only and supports portrait orientation exclusively

### Essential Xcode Commands
- **Build**: Cmd+B
- **Run**: Cmd+R
- **Clean Build Folder**: Cmd+Shift+K
- **Analyze**: Cmd+Shift+B (for static analysis)

## Project Structure

- `Models/`: Data models for User, Meditation, Voice, and PromptSuggestion
- `Views/`: 
  - `Authentication/`: Login, signup, and password reset views
  - `Home/`: Main app interface with meditation history and prompt carousel
  - `Meditation/`: Meditation creation and playback views
  - `Profile/`: User profile and preferences
  - `Shared/`: Reusable UI components (including ErrorView)
- `ViewModels/`: 
  - `AuthViewModel`: Manages authentication state and user sessions
  - `MeditationViewModel`: Orchestrates meditation creation and personalization
  - `AudioPlayerViewModel`: Controls audio playback with AVAudioPlayer
  - `OnboardingViewModel`: Handles onboarding flow state
- `Services/`: 
  - `SupabaseService`: Authentication, database operations, file storage (uses Keychain for secure session storage)
  - `OpenAIService`: AI integrations for script generation (o3-mini), STT (GPT-4o), and TTS (concurrent chunk processing)
  - `StoreManager`: StoreKit 2 integration for credits/subscriptions (handles transaction verification)
- `Helpers/`: Utility functions, extensions, and configuration
- `Database/`: SQL scripts for Supabase setup
- `Supabase/Functions/`: Edge Functions for user registration and App Store notifications
- `Documentation/`: Technical documentation and implementation guides

## Architecture Overview

### Data Flow
1. User input (text/speech) → MeditationViewModel
2. ViewModel → OpenAIService for script generation
3. Script → OpenAIService for TTS conversion (chunked processing)
4. Audio → SupabaseService for storage
5. Meditation record → Supabase database

### Error Handling
- All service methods use Swift's async/await with proper error propagation
- UI displays user-friendly error messages via ErrorView component

## Monetization

The app uses a credit-based system where each credit equals 5 minutes of meditation time. New users receive 1 free credit upon account creation through the `register-user` Edge Function.

### Pricing Structure

**One-Time Purchase:**
- 1 Credit: $0.99 (5 minutes of meditation)

**Subscription Options:**
- Weekly Pass: $4.99/week (10 credits = 50 minutes)
- Monthly Pass: $19.99/month (50 credits = 250 minutes)

Subscriptions offer significant savings (50-60%) compared to individual credit purchases and can be cancelled anytime.

### Monetization Flow
1. User attempts action requiring credits
2. StoreManager checks credit balance
3. If insufficient, StoreView is presented
4. Purchase updates user_credits table via transaction observer
5. Subscriptions auto-renew and add credits monthly/weekly

All purchases are processed through Apple's StoreKit 2 with purchase histories and subscription statuses tracked in the Supabase database.

## Audio Processing

Audio generation follows these steps:

1. Meditation script is generated using OpenAI's o4-mini model
2. Script is sent to ElevenLabs fast api to copnvert to speech 
5. Final audio is stored in Supabase storage and linked to the meditation

## Personalization Features

The app offers intelligent personalization through:

1. **Meditation History Analysis**: The system analyzes up to 7 previous meditation prompts to identify patterns and preferences
2. **One-Click Personalization**: Users can generate personalized meditation prompts with a single tap on the "Personalize" button
3. **Adaptive Suggestions**: The personalization algorithm creates prompts that reflect the user's recurring themes and language patterns
4. **Visual Feedback**: Real-time visual cues inform users when personalized suggestions are being generated
5. **Interactive Prompt Carousel**: Animated carousel displays categorized meditation prompts that users can tap to instantly begin creating a meditation

## Design Philosophy

The app features a calming, blue-themed UI with:

- Time-based ocean gradient backgrounds
- Subtle animated wave overlays
- Consistent typography and spacing
- Optimized for iPhone portrait orientation to provide the best meditation experience

## Testing

### StoreKit Testing
- Use StoreKit Testing in Xcode for IAP development
- Test on physical device for App Store purchases
- **Note**: No automated test suite currently exists

### Simulator Testing Guidelines
When testing in the simulator, create a new profile each time:
- Email format: `ryanatloot+test[NUMBER]@gmail.com` (increment NUMBER each time)
- Next email to use: `ryanatloot+test30@gmail.com`
- Password: Always use `test001!`
- Example sequence: test30, test31, test32, etc.

## Edge Functions

Located in `Supabase/Functions/`:
- **register-user**: Initializes user with 1 free credit on signup
- **app-store-notifications**: Processes Apple server-to-server notifications for subscription updates

Deploy with:
```bash
supabase functions deploy function-name
```

## MCP Server Integration

Claude Code has access to three powerful MCP servers for iOS development:

### 1. Puppeteer (Browser Automation)
- **Tools**: navigate, screenshot, click, fill, select, hover, evaluate
- **Use Cases**: Web scraping, form automation, visual testing, App Store Connect automation

### 2. XcodeBuild (Project Management)
- **Key Tools**:
  - Project discovery: `discover_projs`, `list_schems_ws`, `list_sims`
  - Building: `build_sim_name_ws`, `build_run_sim_name_ws`, `build_dev_ws`
  - Testing: `test_sim_name_ws`, `test_device_ws`
  - UI Automation: `describe_ui`, `tap`, `swipe`, `type_text`, `screenshot`
  - App Management: `install_app_sim`, `launch_app_sim`, `stop_app_sim`

### 3. iOS Simulator
- **Tools**: get_booted_sim_id, ui_describe_all, ui_tap, ui_type, ui_swipe, screenshot, record_video
- **Use Cases**: Accessibility testing, user flow recording, bug reproduction, app demos

### Common MCP Workflows
1. **Testing**: Build app → Boot simulator → Install app → Run UI tests → Take screenshots
2. **Debugging**: Describe UI hierarchy → Tap specific coordinates → Check accessibility
3. **Demo Creation**: Boot simulator → Launch app → Record video → Take screenshots
4. **Performance Testing**: Set network to 3G → Launch app → Monitor performance
