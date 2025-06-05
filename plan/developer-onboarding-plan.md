# Developer Onboarding with MCP Server + RAG Integration

## Executive Summary

This document outlines how to extend our AI architecture with an **MCP server** that enables end-to-end dApp development in developers' local environments (Cursor, VS Code, etc.). The MCP server acts as a bridge between local development tools and our RAG-based AI system.

## Correct Architecture: MCP Server → RAG System

### **The Clean Approach**

```
┌─────────────────────────────────────────────────────────────────┐
│                Developer's Local Environment                    │
│  • Cursor/VS Code with MCP integration                         │
│  • Local file system access                                    │
│  • Terminal and deployment tools                               │
└─────────────────────────────────────────────────────────────────┘
                              │ MCP Protocol
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MCP Server (Local)                         │
│  • Runs on developer's machine                                 │
│  • File system access for scaffolding                          │
│  • Terminal command execution                                  │
│  • API calls to our RAG system                                 │
└─────────────────────────────────────────────────────────────────┘
                              │ HTTP API
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Our RAG AI System                           │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  aptos-docs     │  │  github-bot     │  │   dapp-mcp      │ │
│  │  pipeline       │  │  pipeline       │  │   pipeline      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                  │              │
│                                                  ▼              │
│                            ┌─────────────────────────────────┐  │
│                            │     dApp Development             │  │
│                            │     Knowledge Base              │  │
│                            │  • Aptos Build workflows       │  │
│                            │  • dApp templates              │  │
│                            │  • Best practices              │  │
│                            │  • Code examples               │  │
│                            └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### **Step 1: Create "dapp-mcp" Application Pipeline**

Add a new configuration to our existing AI system:

```json
// configs/dapp-mcp.json
{
  "app_name": "dapp-mcp",
  "version": "1.0",
  "data_sources": [
    {
      "name": "aptos-build-workflows",
      "type": "markdown_docs",
      "path": "data/aptos-build-workflows"
    },
    {
      "name": "dapp-templates",
      "type": "code_repository", 
      "path": "data/dapp-templates"
    },
    {
      "name": "deployment-patterns",
      "type": "markdown_docs",
      "path": "data/deployment-patterns"
    },
    {
      "name": "integration-examples",
      "type": "code_repository",
      "path": "data/integration-examples"
    }
  ],
  "retrieval": {
    "strategy": "multi_step",
    "embedding_model": "openai/text-embedding-3-small",
    "query_expansion_llm": {
      "provider": "openai",
      "model": "gpt-4o-mini",
      "temperature": 0.3
    }
  },
  "llm": {
    "provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.1,
    "max_tokens": 3000,
    "system_prompt_template": "dapp_development_expert",
    "include_sources": true,
    "tools": ["context_search"]
  }
}
```

### **Step 2: MCP Server Implementation**

Create a standalone MCP server that developers install locally:

```typescript
// aptos-dapp-mcp-server/src/server.ts
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';

class AptosDappMCPServer {
  private ragEndpoint = 'https://our-rag-api.com/api/v1/chat';
  
  constructor() {
    this.server = new Server(
      { name: 'aptos-dapp-builder', version: '1.0.0' },
      { capabilities: { tools: {} } }
    );
    
    this.setupTools();
  }
  
  private setupTools() {
    // Tool: Get guidance from RAG system
    this.server.setRequestHandler('tools/call', async (request) => {
      const { name, arguments: args } = request.params;
      
      switch (name) {
        case 'get_dapp_guidance':
          return await this.getDappGuidance(args.query, args.context);
          
        case 'create_dapp_scaffold':
          return await this.createDappScaffold(args.dapp_type, args.features, args.name);
          
        case 'setup_aptos_build_project':
          return await this.setupAptosBuildProject(args.project_name, args.project_type);
          
        case 'deploy_to_testnet':
          return await this.deployToTestnet(args.project_path);
          
        default:
          throw new Error(`Unknown tool: ${name}`);
      }
    });
  }
  
  private async getDappGuidance(query: string, context: any) {
    // Call our RAG system through the dapp-mcp application
    const response = await fetch(this.ragEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        app_name: 'dapp-mcp',
        query,
        params: { context }
      })
    });
    
    const guidance = await response.json();
    return { content: [{ type: 'text', text: guidance.content }] };
  }
  
  private async createDappScaffold(dappType: string, features: string[], name: string) {
    // 1. Get template guidance from RAG
    const guidance = await this.getDappGuidance(
      `Create ${dappType} dApp scaffold with features: ${features.join(', ')}`,
      { action: 'scaffold', dapp_type: dappType, features }
    );
    
    // 2. Execute file creation based on guidance
    const projectPath = `./${name}`;
    await this.createProjectStructure(projectPath, dappType, features);
    
    // 3. Populate files with template content
    await this.populateTemplateFiles(projectPath, dappType, features);
    
    return {
      content: [{
        type: 'text',
        text: `✅ Created ${dappType} dApp scaffold at ${projectPath}\n${guidance.content[0].text}`
      }]
    };
  }
  
  private async createProjectStructure(path: string, dappType: string, features: string[]) {
    const fs = await import('fs/promises');
    
    // Create basic structure
    await fs.mkdir(`${path}/move/sources`, { recursive: true });
    await fs.mkdir(`${path}/frontend/src/components`, { recursive: true });
    await fs.mkdir(`${path}/frontend/src/hooks`, { recursive: true });
    await fs.mkdir(`${path}/deployment`, { recursive: true });
    
    // Additional structure based on dApp type
    if (dappType === 'defi') {
      await fs.mkdir(`${path}/frontend/src/defi`, { recursive: true });
    } else if (dappType === 'nft') {
      await fs.mkdir(`${path}/frontend/src/nft`, { recursive: true });
    } else if (dappType === 'gaming') {
      await fs.mkdir(`${path}/frontend/src/game-engine`, { recursive: true });
    }
  }
  
  private async populateTemplateFiles(path: string, dappType: string, features: string[]) {
    // Get specific file content from RAG system
    const fileQueries = [
      `Generate Move.toml for ${dappType} project`,
      `Generate main smart contract for ${dappType} with features: ${features.join(', ')}`,
      `Generate React frontend setup for ${dappType}`,
      `Generate deployment script for ${dappType}`
    ];
    
    const fileContents = await Promise.all(
      fileQueries.map(query => this.getDappGuidance(query, { 
        action: 'generate_file', 
        dapp_type: dappType, 
        features 
      }))
    );
    
    // Write files based on RAG responses
    const fs = await import('fs/promises');
    await fs.writeFile(`${path}/move/Move.toml`, this.extractCode(fileContents[0]));
    await fs.writeFile(`${path}/move/sources/main.move`, this.extractCode(fileContents[1]));
    await fs.writeFile(`${path}/frontend/src/App.tsx`, this.extractCode(fileContents[2]));
    await fs.writeFile(`${path}/deployment/deploy.ts`, this.extractCode(fileContents[3]));
  }
  
  private async setupAptosBuildProject(projectName: string, projectType: string) {
    // Get setup instructions from RAG
    const guidance = await this.getDappGuidance(
      `Set up Aptos Build project for ${projectType}`,
      { action: 'setup_build_project', project_name: projectName, project_type: projectType }
    );
    
    // Generate .env template
    const envContent = `
# Aptos Build Configuration
APTOS_BUILD_API_KEY=your_api_key_here
APTOS_NETWORK=testnet

# Add your API key from https://build.aptoslabs.com
`;
    
    const fs = await import('fs/promises');
    await fs.writeFile('.env.example', envContent);
    
    return {
      content: [{
        type: 'text',
        text: `✅ Created Aptos Build project setup\n\n${guidance.content[0].text}\n\n📝 Next steps:\n1. Get your API key from https://build.aptoslabs.com\n2. Copy .env.example to .env\n3. Add your API key to .env`
      }]
    };
  }
  
  private extractCode(guidance: any): string {
    // Extract code blocks from RAG response
    const text = guidance.content[0].text;
    const codeBlock = text.match(/```[\s\S]*?```/);
    return codeBlock ? codeBlock[0].replace(/```\w*\n?/g, '').trim() : text;
  }
}

// Start the server
const server = new AptosDappMCPServer();
const transport = new StdioServerTransport();
server.connect(transport);
```

## Developer Workflow

### **Installation:**
```bash
# Install the MCP server
npm install -g @aptos/dapp-mcp-server

# Configure in Cursor/VS Code settings
{
  "mcp.servers": {
    "aptos-dapp-builder": {
      "command": "aptos-dapp-mcp-server",
      "args": []
    }
  }
}
```

### **Usage in Cursor:**
```
Developer: "Create a DeFi swap dApp with gas station support"

Cursor (via MCP): 
→ get_dapp_guidance("DeFi swap architecture with gas station")
→ create_dapp_scaffold(dapp_type="defi", features=["swap", "gas_station"], name="my-swap-dapp")
→ setup_aptos_build_project("my-swap-dapp", "defi")

Result: Complete project created locally with:
- Move contracts for swap functionality
- React frontend with wallet integration  
- Gas station configuration
- Deployment scripts
- Aptos Build integration
```

## Benefits of This Approach

### **✅ Clean Architecture:**
- Our RAG system stays unchanged - just add one more application
- MCP server is a separate, focused component
- No modifications to existing codebase

### **✅ Developer Experience:**
- Works in their familiar environment (Cursor/VS Code)
- Creates files locally where they can edit and customize
- Integrates with their existing git workflow

### **✅ Scalability:**
- Easy to add new MCP tools without touching RAG system
- RAG system handles all the knowledge and guidance
- MCP server handles all the file operations and local actions

### **✅ Flexibility:**
- Developers can use MCP tools selectively
- Can customize generated code immediately
- Works with any MCP-compatible editor

This approach gives us true "vibe code and deploy" capability while maintaining clean separation of concerns and fitting perfectly into our existing architecture!

## Interactive Flowchart Experience

### **Overview: Guided Decision Trees**

To address requirements for common workflow guidance and reduce decision paralysis, we'll add an **interactive flowchart system** that guides users through structured decision trees.

### **Two Main Use Cases:**

1. **Interactive Flowcharts** - Informational guidance for common workflows (RAG only)
2. **End-to-End dApp Creation** - Actionable dApp building with file creation (RAG + MCP)

### **Architecture: Two Different UIs, Same Backend**

```
┌─────────────────────────────────────────────────────────────────┐
│              Interactive Flowchart UI (Web)                     │
│  • Visual decision trees                                       │
│  • Step-by-step instructions                                   │
│  • Code examples and best practices                            │
│  • NO file creation - just guidance                            │
└─────────────────────────────────────────────────────────────────┘
                              │ RAG API calls only
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Our RAG AI System                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  flowchart      │  │   dapp-mcp      │  │  other apps     │ │
│  │  guide          │  │   pipeline      │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ RAG + MCP calls
┌─────────────────────────────────────────────────────────────────┐
│              End-to-End dApp Creation (MCP)                     │
│  • Runs locally in Cursor/VS Code                              │
│  • Creates actual files and projects                           │
│  • Executes deployment actions                                 │
│  • Calls RAG for guidance + MCP for actions                    │
└─────────────────────────────────────────────────────────────────┘
```

### **Flowchart Data Structure**

Add a new application to our RAG system for flowchart management:

```json
// configs/flowchart-guide.json
{
  "app_name": "flowchart-guide",
  "version": "1.0",
  "data_sources": [
    {
      "name": "flowchart-definitions",
      "type": "structured_data",
      "path": "data/flowcharts"
    },
    {
      "name": "workflow-steps",
      "type": "markdown_docs",
      "path": "data/workflow-steps"
    },
    {
      "name": "decision-logic",
      "type": "structured_data",
      "path": "data/decision-trees"
    }
  ],
  "retrieval": {
    "strategy": "simple",
    "embedding_model": "openai/text-embedding-3-small"
  },
  "llm": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.1,
    "system_prompt_template": "flowchart_guide_assistant",
    "tools": ["context_search"]
  }
}
```

### **Example Flowchart: End-to-End dApp Creation**

```yaml
# data/flowcharts/dapp-creation-flow.yaml
name: "Build Your First dApp"
description: "Complete guided experience from concept to deployment"
start_node: "project_type"

nodes:
  project_type:
    type: "choice"
    question: "What type of dApp do you want to build?"
    description: "Choose the category that best fits your project vision"
    options:
      - label: "DeFi (Trading, Swapping, Lending)"
        value: "defi"
        next: "defi_features"
        icon: "💰"
      - label: "NFT (Marketplace, Collection, Gaming)"
        value: "nft" 
        next: "nft_features"
        icon: "🎨"
      - label: "Gaming (P2E, Battle, Strategy)"
        value: "gaming"
        next: "gaming_features"
        icon: "🎮"
      - label: "DAO (Governance, Voting)"
        value: "dao"
        next: "dao_features"
        icon: "🏛️"
  
  defi_features:
    type: "multi_choice"
    question: "Which DeFi features do you need?"
    description: "Select all features you want to include"
    options:
      - label: "Token Swapping"
        value: "swap"
        required_for: ["liquidity_pools"]
      - label: "Liquidity Pools" 
        value: "liquidity"
        requires: ["swap"]
      - label: "Staking/Farming"
        value: "staking"
      - label: "Lending/Borrowing"
        value: "lending"
      - label: "Gas Sponsorship"
        value: "gas_station"
    next: "user_onboarding"
    
  user_onboarding:
    type: "choice"
    question: "How should users connect to your dApp?"
    options:
      - label: "Traditional Wallets Only (Petra, Martian)"
        value: "wallet_only"
        next: "data_needs"
      - label: "Social Login + Wallets (Easier for new users)"
        value: "social_and_wallet"
        next: "social_providers"
      - label: "Social Login Only (Web2 UX)"
        value: "social_only"
        next: "social_providers"
        
  social_providers:
    type: "multi_choice"
    question: "Which social login providers?"
    options:
      - label: "Google"
        value: "google"
      - label: "Discord"
        value: "discord"
      - label: "Apple"
        value: "apple"
      - label: "Twitter/X"
        value: "twitter"
    next: "data_needs"
    
  data_needs:
    type: "choice"
    question: "What data will your dApp need to track?"
    description: "This helps us configure your no-code indexer"
    options:
      - label: "Basic (User balances, transactions)"
        value: "basic"
        next: "deployment_target"
      - label: "Advanced (Analytics, leaderboards, history)"
        value: "advanced" 
        next: "analytics_config"
      - label: "Custom (I'll configure it myself)"
        value: "custom"
        next: "deployment_target"
        
  deployment_target:
    type: "choice"
    question: "Where do you want to deploy?"
    options:
      - label: "Testnet First (Recommended)"
        value: "testnet"
        next: "create_project"
      - label: "Mainnet (Production)"
        value: "mainnet"
        next: "mainnet_warning"
        
  create_project:
    type: "action"
    action: "create_dapp_scaffold"
    title: "Creating Your dApp..."
    description: "Generating project structure, smart contracts, and frontend"
    mcp_call: true
    next: "setup_build"
    
  setup_build:
    type: "action" 
    action: "setup_aptos_build_project"
    title: "Setting Up Aptos Build Integration..."
    description: "Configuring API access, gas station, and indexer"
    mcp_call: true
    next: "completion"
    
  completion:
    type: "success"
    title: "🎉 Your dApp is Ready!"
    description: "Your project has been created and configured"
    next_steps:
      - "Review the generated code in your project folder"
      - "Customize the smart contracts for your needs"
      - "Test the frontend with your wallet"
      - "Deploy to testnet when ready"
    actions:
      - label: "Open in Cursor"
        action: "open_in_cursor"
      - label: "Deploy Now"
        action: "deploy_to_testnet"
      - label: "View Tutorial"
        action: "show_tutorial"
```

### **Example Flowchart: Common Workflow - API Key Creation**

```yaml
# data/flowcharts/api-key-workflow.yaml
name: "Create Aptos Build API Key"
description: "Step-by-step guide to generate and configure your API key"
start_node: "check_account"

nodes:
  check_account:
    type: "choice"
    question: "Do you have an Aptos Build account?"
    options:
      - label: "Yes, I'm already signed up"
        value: "has_account"
        next: "project_type"
      - label: "No, I need to create one"
        value: "needs_account"
        next: "create_account"
        
  create_account:
    type: "instruction"
    title: "Create Your Aptos Build Account"
    steps:
      - "Go to https://build.aptoslabs.com"
      - "Click 'Sign Up' in the top right"
      - "Use your email or GitHub account"
      - "Verify your email address"
    next: "project_type"
    
  project_type:
    type: "choice"
    question: "What type of project is this API key for?"
    description: "This helps us set the right permissions"
    options:
      - label: "DeFi Application"
        value: "defi"
        next: "defi_permissions"
      - label: "NFT Project"
        value: "nft"
        next: "nft_permissions"
      - label: "Gaming dApp"
        value: "gaming"
        next: "gaming_permissions"
      - label: "Development/Learning"
        value: "development"
        next: "dev_permissions"
      - label: "General Purpose"
        value: "general"
        next: "general_permissions"
        
  defi_permissions:
    type: "info"
    title: "DeFi Project API Permissions"
    content: "For DeFi applications, you'll need:\n• Node API access for transactions\n• Indexer API for swap/liquidity data\n• Higher rate limits for trading volume"
    next: "create_key_steps"
    
  create_key_steps:
    type: "instruction"
    title: "Creating Your API Key"
    steps:
      - "Log into https://build.aptoslabs.com"
      - "Click 'Create New Project'"
      - "Enter project name and select 'DeFi' category"
      - "Click 'Generate API Key'"
      - "Copy and securely save your API key"
    code_example: |
      ```bash
      # Add to your .env file
      APTOS_BUILD_API_KEY=your_api_key_here
      APTOS_NETWORK=testnet
      ```
    next: "integration_code"
    
  integration_code:
    type: "code"
    title: "Integrate API Key in Your Code"
    description: "Here's how to use your API key with the Aptos SDK"
    code_example: |
      ```typescript
      import { AptosConfig, Aptos, Network } from "@aptos-labs/ts-sdk";
      
      const config = new AptosConfig({
        network: Network.TESTNET,
        clientConfig: {
          API_KEY: process.env.APTOS_BUILD_API_KEY
        }
      });
      
      const aptos = new Aptos(config);
      
      // Test your connection
      async function testConnection() {
        try {
          const ledgerInfo = await aptos.getLedgerInfo();
          console.log("✅ Connected to Aptos!", ledgerInfo);
        } catch (error) {
          console.error("❌ Connection failed:", error);
        }
      }
      ```
    next: "test_connection"
    
  test_connection:
    type: "instruction"
    title: "Test Your API Connection"
    steps:
      - "Run the test code above in your project"
      - "Check your console for the success message"
      - "If it fails, verify your API key is correct"
      - "Check the Aptos Build dashboard for usage stats"
    next: "success"
    
  success:
    type: "success"
    title: "✅ API Key Ready!"
    description: "Your API key is configured and tested"
    next_steps:
      - "Your API key is working correctly"
      - "View usage in the Aptos Build dashboard"
      - "Ready to build your dApp!"
    resources:
      - label: "View Usage Dashboard"
        url: "https://build.aptoslabs.com/dashboard"
      - label: "API Documentation"
        url: "https://developers.aptoslabs.com/docs"
      - label: "Code Examples"
        url: "https://github.com/aptos-labs/aptos-ts-sdk/tree/main/examples"
```

### **Example Flowchart: Gas Station Setup**

```yaml
# data/flowcharts/gas-station-workflow.yaml
name: "Set Up Gas Station"
description: "Configure sponsored transactions to improve user experience"
start_node: "understand_gas_station"

nodes:
  understand_gas_station:
    type: "info"
    title: "What is Gas Station?"
    content: |
      Gas Station lets you sponsor transaction fees for your users, creating a smoother onboarding experience.
      
      ✅ **Good for:**
      • New user onboarding
      • Small transactions (voting, claiming)
      • Promotional campaigns
      
      ❌ **Avoid for:**
      • Large financial transactions
      • Bot transactions
      • High-frequency trading
    next: "check_eligibility"
    
  check_eligibility:
    type: "choice"
    question: "What best describes your use case?"
    options:
      - label: "New user onboarding (first few transactions)"
        value: "onboarding"
        next: "onboarding_setup"
      - label: "Small operations (voting, claiming, profiles)"
        value: "small_ops"
        next: "small_ops_setup"
      - label: "Large transactions or trading"
        value: "large_transactions"
        next: "not_recommended"
        
  onboarding_setup:
    type: "info"
    title: "Onboarding Gas Station Strategy"
    content: |
      **Recommended Setup:**
      • Limit: 5 transactions per new user
      • Max gas per transaction: 1000 units
      • Allowed functions: wallet_connect, profile_setup, first_nft_mint
      • Reset period: 30 days
    next: "implementation_steps"
    
  implementation_steps:
    type: "instruction"
    title: "Implementation Steps"
    steps:
      - "Go to your Aptos Build dashboard"
      - "Navigate to Gas Station section"
      - "Click 'Create Gas Station'"
      - "Set funding amount (start with 10 APT for testing)"
      - "Configure sponsorship rules as shown above"
      - "Deploy to testnet first"
    next: "integration_code"
    
  integration_code:
    type: "code"
    title: "Integrate Gas Station in Your dApp"
    code_example: |
      ```typescript
      import { AptosConfig, Aptos, Account } from "@aptos-labs/ts-sdk";
      
      class GasStationManager {
        private gasStationAccount: Account;
        private aptos: Aptos;
        
        constructor(gasStationPrivateKey: string) {
          this.gasStationAccount = Account.fromPrivateKey({
            privateKey: gasStationPrivateKey
          });
          
          this.aptos = new Aptos(new AptosConfig({
            network: Network.TESTNET,
            clientConfig: {
              API_KEY: process.env.APTOS_BUILD_API_KEY
            }
          }));
        }
        
        async shouldSponsorTransaction(
          userAddress: string, 
          functionName: string
        ): Promise<boolean> {
          // Check if user is eligible for sponsorship
          const userTransactionCount = await this.getUserTransactionCount(userAddress);
          
          // Sponsor first 5 transactions for new users
          if (userTransactionCount < 5) {
            return ['connect_wallet', 'setup_profile', 'mint_starter_nft']
              .includes(functionName);
          }
          
          return false;
        }
        
        async sponsorTransaction(transaction: any) {
          // Use gas station account as fee payer
          const sponsoredTxn = await this.aptos.transaction.build.multiAgent({
            sender: transaction.sender,
            secondarySigners: [this.gasStationAccount.accountAddress],
            data: transaction.data
          });
          
          return sponsoredTxn;
        }
      }
      ```
    next: "testing_guide"
    
  testing_guide:
    type: "instruction"
    title: "Test Your Gas Station"
    steps:
      - "Create a test user account"
      - "Try a sponsored transaction (like profile setup)"
      - "Verify the transaction was sponsored (user paid no gas)"
      - "Check gas station balance in dashboard"
      - "Monitor for abuse patterns"
    next: "monitoring"
    
  monitoring:
    type: "info"
    title: "Monitor and Optimize"
    content: |
      **Key Metrics to Track:**
      • Daily sponsored transaction count
      • Gas station balance usage
      • User conversion (sponsored → paying users)
      • Abuse patterns (same user, rapid transactions)
      
      **Optimization Tips:**
      • Start conservative, increase limits based on data
      • Set daily/weekly caps per user
      • Monitor for bot activity
      • Adjust sponsored function list based on usage
    next: "success"
    
  success:
    type: "success"
    title: "🎉 Gas Station Active!"
    description: "Your gas station is configured and ready to sponsor transactions"
    next_steps:
      - "Monitor usage in Aptos Build dashboard"
      - "Track user conversion metrics"
      - "Adjust limits based on real usage patterns"
    resources:
      - label: "Gas Station Dashboard"
        url: "https://build.aptoslabs.com/gas-station"
      - label: "Sponsorship Best Practices"
        url: "https://developers.aptoslabs.com/docs/gas-station"
```

### **User Experience Flow**

#### **Web Interface:**
```typescript
// Flowchart Web Component
function FlowchartGuide({ flowchartName }: { flowchartName: string }) {
  const [currentNode, setCurrentNode] = useState('start');
  const [userChoices, setUserChoices] = useState({});
  const [flowchart, setFlowchart] = useState(null);
  
  // Load flowchart from RAG system
  useEffect(() => {
    fetch('/api/v1/chat', {
      method: 'POST',
      body: JSON.stringify({
        app_name: 'flowchart-guide',
        query: `Get flowchart definition for ${flowchartName}`,
        params: { flowchart_name: flowchartName }
      })
    }).then(res => res.json())
      .then(data => setFlowchart(data.flowchart));
  }, [flowchartName]);
  
  const handleChoice = async (choice: string) => {
    const node = flowchart.nodes[currentNode];
    const newChoices = { ...userChoices, [currentNode]: choice };
    setUserChoices(newChoices);
    
    // Execute action if this is an action node
    if (node.type === 'action' && node.mcp_call) {
      await executeMCPAction(node.action, newChoices);
    }
    
    // Move to next node
    const nextNode = node.options?.find(opt => opt.value === choice)?.next || node.next;
    setCurrentNode(nextNode);
  };
  
  return (
    <div className="flowchart-container">
      <FlowchartNode 
        node={flowchart?.nodes[currentNode]}
        onChoice={handleChoice}
        progress={calculateProgress(currentNode, flowchart)}
      />
    </div>
  );
}
```

#### **Integration with MCP:**
```typescript
// Enhanced MCP Server with Flowchart Support
class AptosDappMCPServer {
  // ... existing code ...
  
  private async executeFlowchartAction(action: string, context: any) {
    switch (action) {
      case 'create_dapp_scaffold':
        return await this.createDappScaffold(
          context.project_type, 
          context.features || [], 
          context.project_name || 'my-dapp'
        );
        
      case 'guide_api_key_creation':
        // Open browser to Aptos Build dashboard
        const { exec } = await import('child_process');
        exec('open https://build.aptoslabs.com/projects/new');
        
        return {
          content: [{
            type: 'text',
            text: '🌐 Opened Aptos Build dashboard. Follow these steps:\n1. Click "Create New Project"\n2. Enter your project details\n3. Click "Generate API Key"\n4. Copy the key and return here'
          }]
        };
        
      case 'setup_api_key_locally':
        // Get API key from user and set up .env
        const apiKey = context.api_key || process.env.APTOS_BUILD_API_KEY;
        await this.setupEnvironmentFile(apiKey, context.project_type);
        
        return {
          content: [{
            type: 'text', 
            text: '✅ API key configured in .env file'
          }]
        };
    }
  }
}
```

### **Benefits of Flowchart Approach**

#### **✅ Reduces Decision Paralysis**
- Users don't need to know what questions to ask
- Clear, structured progression through complex decisions
- Visual progress tracking

#### **✅ Educational Value**
- Users learn best practices while building
- Understand why certain choices matter
- See the full development lifecycle

#### **✅ Addresses Leadership Requirements**
- **Common Workflows**: Pre-built flows for API keys, gas station, wallet setup, etc.
- **Consistent Guidance**: Ensures all users follow best practices
- **Scalable**: Easy to add new workflows based on GitHub questions

#### **✅ Flexible Integration**
- **Web Interface**: Guided experience for beginners
- **MCP Integration**: Execute steps locally in Cursor/VS Code
- **RAG Powered**: All guidance comes from our knowledge base



This gives us both the guided experience for beginners and the power tools for experienced developers!
