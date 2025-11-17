'use strict';

class NovioClient {
    constructor() {
        this.Connected = new Promise((resolve) => {
            document.addEventListener('onNovioConnected', function (event) {
                window.novio.isConnected = true;
                resolve();
            });
        });
        this.Disconnected = new Promise((resolve) => {
            document.addEventListener('onNovioDisconnected', function (event) {
                window.novio.isConnected = false;
                resolve();

                const novioConnect = document.querySelector('novio-connect')
                if (novioConnect != null) {
                    novioConnect.disconnect();
                }
            });
        });

        this.SignedIn = new Promise((resolve) => {
            document.addEventListener('NovioSignInSuccess', function (event) {
                resolve();
            });
        });

        let clientDisconnectedResolve;
        this.ClientDisconnected = new Promise((resolve) => {
            clientDisconnectedResolve = resolve;
        });

        this.ClientConnected = new Promise((resolve) => {
            document.addEventListener('NovioClientConnected', function (event) {
                resolve();

                const novioConnect = document.querySelector('novio-connect')
                if (novioConnect != null) {
                    novioConnect.setLiveConnectionState('live');
                }

                // Only listen for disconnect after connected
                document.addEventListener('NovioClientDisconnected', function (event) {
                    novioConnect.setLiveConnectionState('disconnected');
                    clientDisconnectedResolve();
                }, { once: true });
            }, { once: true });
        });

        // Initialize pending requests map for send functionality
        this.pendingRequests = new Map();

        // Initialize message handlers array
        this.messageHandlers = [];

        // Set up all event listeners
        this._setupEventListeners();
    };

    transferTo(toAddress, amount, attrs) {
        return this.sendTransactionRequest('transferTo', [toAddress, amount, attrs]);
    }

    registerName(name, attrs) {
        return this.sendTransactionRequest('registerName', [name, attrs]);
    }

    transferName(name, recipient, attrs) {
        return this.sendTransactionRequest('transferName', [name, recipient, attrs]);
    }

    deleteName(name, attrs) {
        return this.sendTransactionRequest('deleteName', [name, attrs]);
    }

    subscribe(topic, duration, identifier, meta, attrs) {
        return this.sendTransactionRequest('subscribe', [topic, duration, identifier, meta, attrs]);
    }

    unsubscribe(topic, identifier, attrs) {
        return this.sendTransactionRequest('unsubscribe', [topic, identifier, attrs]);
    }

    sign(message) {
        return this.sendSignatureRequest(message);
    }

    sendTransactionRequest(type, data) {
        const txRequestEvent = new CustomEvent("onNovioTxRequest", {
            bubbles: true,
            cancelable: false,
            detail: {
                type: type,
                data: data
            },
        });
        document.dispatchEvent(txRequestEvent);

        return new Promise((resolve, reject) => {
            window.addEventListener("onNovioTxResponse", (event) => {
                if (event.detail.data.error) {
                    reject(event.detail.data.error);
                } else {
                    resolve(event.detail);
                }
            }, false);
        })
    }

    sendSignatureRequest(message) {
        const signRequestEvent = new CustomEvent("onNovioSignRequest", {
            bubbles: true,
            cancelable: false,
            detail: {
                data: message,
                allowClient: true,
            },
        });
        document.dispatchEvent(signRequestEvent);

        return new Promise((resolve, reject) => {
            window.addEventListener("onNovioSignResponse", (event) => {
                if (event.detail.data.error) {
                    reject(event.detail.data.error);
                } else {
                    resolve(event.detail);
                }
            }, false);
        })
    }

    /**
     * Sets up all event listeners for the SDK
     * @private
     */
    _setupEventListeners() {
        // Handle client messages (both replies and regular messages)
        document.addEventListener('onNovioClientMessage', (msg) => {
            // Check if this is a reply to a pending request
            if (msg.detail.message.replyTo != undefined) {
                const pending = this.pendingRequests.get(msg.detail.message.replyTo);

                if (pending) {
                    this.pendingRequests.delete(msg.detail.message.replyTo);
                    if (msg.detail.message.Message != undefined) {
                        pending.resolve(msg.detail.message.Message);
                    } else {
                        pending.reject(new Error('No message in response'));
                    }
                }
            } else {
                // Handle regular incoming messages by calling registered handlers
                this.messageHandlers.forEach(handler => {
                    handler({
                        src: msg.detail.message.Sender,
                        payload: msg.detail.message.Message
                    });
                });
            }
        });

        // Handle ping events
        document.addEventListener('onNovioClientPing', (msg) => {
            // Ping events can be handled here if needed
            // For now, just a placeholder for future functionality
            // The extension pings to keep alive the background-foreground connection.
        });
    }

    /**
     * Registers a callback for incoming messages
     * @param {function} handler - Function with signature ({ src, payload })
     */
    onMessage(handler) {
        this.messageHandlers.push(handler);
    }

    /**
     * Sends a message with advanced options including timeout handling
     * @param {string} addr - The address to send to
     * @param {string|object} payload - The message payload (string or object)
     * @param {object} [options] - Send options
     * @param {boolean} [options.noReply=false] - If true, don't wait for a reply
     * @param {number} [options.responseTimeout=5000] - Timeout in milliseconds for replies
     * @param {string} [requestId] - Optional custom request ID
     * @returns {Promise|undefined} - Returns Promise if expecting reply, undefined for fire-and-forget
     */
    send(addr, payload, options = {}, requestId) {
        const defaultOptions = {
            noReply: false,
            responseTimeout: 60000
        };

        options = { ...defaultOptions, ...options };
        const id = requestId || crypto.randomUUID();

        // Convert payload to string if it's an object
        const processedPayload = typeof payload === 'string' ? payload : JSON.stringify(payload);

        // Dispatch the send event
        const clientSendEvent = new CustomEvent("onNovioClientSend", {
            bubbles: true,
            cancelable: false,
            detail: {
                data: { addr: addr, payload: processedPayload, options: options, requestId: id }
            },
        });
        document.dispatchEvent(clientSendEvent);

        // Return undefined for fire-and-forget messages
        if (options.noReply) {
            return undefined;
        }

        // Return Promise for messages expecting replies
        return new Promise((resolve, reject) => {
            this.pendingRequests.set(id, { resolve, reject });

            // Set up timeout
            setTimeout(() => {
                if (this.pendingRequests.has(id)) {
                    this.pendingRequests.delete(id);
                    reject(new Error('Request timeout'));
                }
            }, options.responseTimeout);
        });
    }

    async getSubscribers(topic, options) {
        return {
            subscribers: [],
            subscribersInTxPool: [],
        };
    }

    hexToBytes(hex) {
        const bytes = new Uint8Array(hex.length / 2);
        for (let i = 0; i < hex.length; i += 2) {
            bytes[i / 2] = parseInt(hex.substr(i, 2), 16);
        }
        return bytes;
    }

    async messageToSignable(message) {
        const prefixedMsg = `NKN Signed Message:\n${message}`;
        const encoder = new TextEncoder();
        const encodedMsg = encoder.encode(prefixedMsg);

        const hashedOnce = await crypto.subtle.digest('SHA-256', encodedMsg);
        const hashedTwice = await crypto.subtle.digest('SHA-256', hashedOnce);
        return new Uint8Array(hashedTwice);
    }

    async verifyMessage(message, signature, publicKey) {
        try {
            const challenge = await this.messageToSignable(message);
            const signatureBytes = this.hexToBytes(signature);
            const publicKeyBytes = this.hexToBytes(publicKey);

            const cryptoKey = await crypto.subtle.importKey(
                'raw',
                publicKeyBytes,
                {
                    name: 'Ed25519',
                    namedCurve: 'Ed25519'
                },
                false,
                ['verify']
            );

            const isValid = await crypto.subtle.verify(
                'Ed25519',
                cryptoKey,
                signatureBytes,
                challenge
            );

            return isValid;
        } catch (error) {
            console.error('Verification error:', error);
            return false;
        }
    }
}

class NovioWalletButton extends HTMLElement {

    isConnecting = false;
    isConnected = false;

    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.isConnecting = false;
        this.isConnected = false;
        this.wallet = null;

        // Store bound event handlers for cleanup
        this.boundHandlers = {
            connectClick: null,
            documentClick: null,
            menuClick: null,
            signOutClick: null
        };

        // Configuration options
        this.config = {
            theme: this.getAttribute('theme') || 'default',
            size: this.getAttribute('size') || 'medium',
        };

        this.render();
        // Bind events after render is complete
        setTimeout(() => this.bindEvents(), 0);
    }

    static get observedAttributes() {
        return ['theme', 'size'];
    }

    attributeChangedCallback(name, oldValue, newValue) {
        if (oldValue !== newValue) {
            this.config[name.replace('-', '')] = newValue;
            this.render();
            // Re-bind events after re-render
            setTimeout(() => this.bindEvents(), 0);
        }
    }

    render() {
        const { theme, size } = this.config;

        this.shadowRoot.innerHTML = `
      <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&display=swap');
        
        :host {
          display: inline-block;
          font-family: 'Radio Canada', 'Source Sans Pro', sans-serif;
          --btn-bg: var(--accent, #4b5563);
          --btn-text: var(--text-on-accent, #ffffff);
          --btn-border: var(--border-subtle, #dcdcdc);
          --bg-primary: #f5f5f5;
          --bg-secondary: #e0e0e0;
          --text-primary: var(--text-main, #1f2933);
          --text-secondary: #6b7280;
          --text-muted: #657b83;
          --accent-primary: var(--accent, #4b5563);
          --accent-secondary: #cb4b16;
          --success: #16a34a;
          --success-secondary: #2aa198;
          --error: #dc322f;
          --border-color: var(--border-subtle, rgba(0, 0, 0, 0.08));
        }

        .novio-connect {
          position: relative;
          display: inline-block;
        }

        .connect-button {
          padding: ${size === 'small' ? '6px 12px' : size === 'large' ? '12px 20px' : '8px 16px'};
          background: var(--btn-bg);
          border: 1px solid var(--btn-border);
          border-radius: var(--radius-inner, 8px);
          color: var(--btn-text);
          font-family: 'Radio Canada', 'Source Sans Pro', sans-serif;
          font-size: ${size === 'small' ? '13px' : size === 'large' ? '16px' : '14px'};
          font-weight: 600;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: ${size === 'small' ? '8px' : '12px'};
          transition: all 0.3s ease;
          position: relative;
          overflow: hidden;
          min-width: ${size === 'small' ? '120px' : size === 'large' ? '170px' : '140px'};
        }

        .connect-button::before {
          content: none;
        }

        .connect-button:hover:not(:disabled) {
          box-shadow: 0 6px 14px rgba(0, 0, 0, 0.08);
        }

        .connect-button:active:not(:disabled) {
          transform: translateY(0);
        }

        .connect-button:disabled {
          opacity: 0.7;
          cursor: not-allowed;
        }

        .button-icon {
          width: ${size === 'small' ? '16px' : '20px'};
          height: ${size === 'small' ? '16px' : '20px'};
          fill: currentColor;
        }

        .connected {
          background: var(--btn-bg) !important;
        }

        .error {
          background: linear-gradient(135deg, var(--error) 0%, var(--accent-secondary) 100%) !important;
        }

        .spinner {
          animation: spin 1s linear infinite;
        }

        .live-indicator {
            display: none;
            width: 6px;
            height: 6px;
            background: var(--accent-primary);
            border-radius: 50%;
            animation: opacityPulse 1.5s infinite;
        }

        .live-indicator.disconnected {
            background: var(--error);
            animation: none !important;
        }

        .live-indicator.connecting {
            background: var(--success, #ffa500);
            animation: fall 2s infinite ease-in-out;
        }

        @keyframes fall {
            0% {
                transform: translateY(-100%);
                opacity: 0;
            }
            10% {
                opacity: 1;
            }
            90% {
                opacity: 1;
            }
            100% {
                transform: translateY(+100%);
                opacity: 0;
            }
        }

        /* Dropdown menu */
        .dropdown-menu {
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
           
            background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            border: var(--accent-primary);
            border-width: 1px;
            color: var(--text-primary);
            font-family: 'Source Sans Pro', sans-serif;
            font-size: ${size === 'small' ? '14px' : size === 'large' ? '18px' : '16px'};
            font-weight: 600;

            z-index: 1000;
            opacity: 0;
            visibility: hidden;
            transform: translateY(-10px);
            transition: all 0.2s ease;
        }

        .dropdown-menu.open {
            opacity: 1;
            visibility: visible;
            transform: translateY(0);
        }

        @keyframes opacityPulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }

        @keyframes pulse {
          0% { box-shadow: 0 0 0 0 rgba(220, 50, 47, 0.7); }
          70% { box-shadow: 0 0 0 6px rgba(220, 50, 47, 0); }
          100% { box-shadow: 0 0 0 0 rgba(220, 50, 47, 0); }
        }

        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        /* Theme variations */
        :host([theme="minimal"]) .connect-button {
          background: transparent;
          border: 2px solid var(--accent-primary);
          color: var(--accent-primary);
          box-shadow: none;
        }

        :host([theme="minimal"]) .connect-button:hover:not(:disabled) {
          background: var(--accent-primary);
          color: var(--text-primary);
        }

        :host([theme="dark"]) {
          --bg-primary: #1a1a1a;
          --bg-secondary: #2d2d2d;
          --text-primary: #ffffff;
          --border-color: rgba(181, 137, 0, 0.3);
        }
      </style>

      <div class="novio-connect">
        <button class="connect-button" id="connectButton">
          <svg class="button-icon" id="buttonIcon" viewBox="0 0 24 24">
            <path d="M21 18v1c0 1.1-.9 2-2 2H5c-1.11 0-2-.9-2-2V5c0-1.1.89-2 2-2h14c1.1 0 2 .9 2 2v1h-9c-1.11 0-2 .9-2 2v8c0 1.1.89 2 2 2h9zm-9-2h10V8H12v8zm4-2.5c-.83 0-1.5-.67-1.5-1.5s.67-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5z"/>
          </svg>
          <span id="buttonText">Novio Login</span>
          <span class="live-indicator" id="liveIndicator"></span>
          <div style="font-size: 9px; display:none;" id="liveIndicatorText">live</div>
        </button>
        <div class="dropdown-menu" id="dropdownMenu">
            <button class="connect-button" style="width:100%;" id="signOutButton">sign out</button>
        </div>
      </div>
    `;
    }

    // Clean up all event listeners before binding new ones
    unbindEvents() {
        const button = this.shadowRoot.getElementById('connectButton');
        const menu = this.shadowRoot.getElementById('dropdownMenu');
        const signOutButton = this.shadowRoot.getElementById('signOutButton');

        // Remove connect button listener
        if (this.boundHandlers.connectClick && button) {
            button.removeEventListener('click', this.boundHandlers.connectClick);
        }

        // Remove document click listener
        if (this.boundHandlers.documentClick) {
            document.removeEventListener('click', this.boundHandlers.documentClick);
        }

        // Remove menu click listener
        if (this.boundHandlers.menuClick && menu) {
            menu.removeEventListener('click', this.boundHandlers.menuClick);
        }

        // Remove sign out button listener
        if (this.boundHandlers.signOutClick && signOutButton) {
            signOutButton.removeEventListener('click', this.boundHandlers.signOutClick);
        }

        // Clear onclick handlers
        if (button) button.onclick = null;
        if (signOutButton) signOutButton.onclick = null;

        // Reset bound handlers
        this.boundHandlers = {
            connectClick: null,
            documentClick: null,
            menuClick: null,
            signOutClick: null
        };
    }

    bindEvents() {
        // Always clean up first to prevent duplicates
        this.unbindEvents();

        const button = this.shadowRoot.getElementById('connectButton');
        if (!button) return;

        // Bind connect button click
        this.boundHandlers.connectClick = () => this.handleConnect();
        button.addEventListener('click', this.boundHandlers.connectClick);
    }

    bindDropdownEvents() {
        const button = this.shadowRoot.getElementById('connectButton');
        const menu = this.shadowRoot.getElementById('dropdownMenu');
        const signOutButton = this.shadowRoot.getElementById('signOutButton');

        if (!button || !menu || !signOutButton) return;

        let isOpen = false;

        const toggleDropdown = () => {
            if (isOpen) {
                closeDropdown();
            } else {
                openDropdown();
            }
        };

        const openDropdown = () => {
            menu.classList.add('open');
            button.classList.add('active');
            isOpen = true;
        };

        const closeDropdown = () => {
            menu.classList.remove('open');
            button.classList.remove('active');
            isOpen = false;
        };

        // Bind dropdown toggle (replace the connect handler)
        this.unbindEvents(); // Clean up first
        this.boundHandlers.connectClick = (e) => {
            e.stopPropagation();
            toggleDropdown();
        };
        button.addEventListener('click', this.boundHandlers.connectClick);

        // Bind document click to close dropdown
        this.boundHandlers.documentClick = (e) => {
            if (!menu.contains(e.target) && !button.contains(e.target)) {
                closeDropdown();
            }
        };
        document.addEventListener('click', this.boundHandlers.documentClick);

        // Bind menu click to prevent closing
        this.boundHandlers.menuClick = (e) => {
            e.stopPropagation();
        };
        menu.addEventListener('click', this.boundHandlers.menuClick);

        // Bind sign out button
        this.boundHandlers.signOutClick = () => {
            if (window.novio && window.novio.novioSignOut) {
                window.novio.novioSignOut();
            }
            this.reset();
        };
        signOutButton.addEventListener('click', this.boundHandlers.signOutClick);
    }

    async handleConnect() {
        if (this.isConnecting || this.isConnected) return;

        try {
            this.setConnectingState();
            // Check if Novio wallet is installed
            if (typeof window.novio === 'undefined' || !window.novio.isInstalled) {
                throw new Error('WALLET_NOT_INSTALLED');
            }

            var useClient = document.querySelector('novio-connect').getAttribute('useClient');
            var useTls = document.querySelector('novio-connect').getAttribute('useTls');
            var useMessageEncryption = document.querySelector('novio-connect').getAttribute('useMessageEncryption');
            var clientIdentifier = document.querySelector('novio-connect').getAttribute('clientIdentifier');

            const isValid = await window.novio.novioSignIn(useClient, useTls, useMessageEncryption, clientIdentifier);
            if (!isValid) {
                throw new Error('SIGNATURE_FAILED');
            } else {
                this.setConnectedState();


                const novioConnect = document.querySelector('novio-connect')
                if (novioConnect != null) {
                    novioConnect.setLiveConnectionState('connecting');
                }
            }

        } catch (error) {
            this.setErrorState(error);
            this.dispatchEvent(new CustomEvent('novio-error', {
                detail: { error: error.message },
                bubbles: true
            }));
        }
    }

    setConnectingState() {
        this.isConnecting = true;
        const button = this.shadowRoot.getElementById('connectButton');
        const buttonText = this.shadowRoot.getElementById('buttonText');
        const buttonIcon = this.shadowRoot.getElementById('buttonIcon');

        if (button) button.disabled = true;
        if (buttonText) buttonText.textContent = 'Connecting...';
        if (buttonIcon) buttonIcon.classList.add('spinner');
    }

    setConnectedState() {
        this.isConnecting = false;
        this.isConnected = true;

        const button = this.shadowRoot.getElementById('connectButton');
        const buttonText = this.shadowRoot.getElementById('buttonText');
        const buttonIcon = this.shadowRoot.getElementById('buttonIcon');

        if (button) {
            button.disabled = false;
            button.classList.add('connected');
        }

        if (buttonText && window.novio && window.novio.address) {
            buttonText.textContent = window.novio.address.substring(0, 6) + '...' +
                window.novio.address.substring(window.novio.address.length - 6);
        }

        if (buttonIcon) {
            buttonIcon.classList.remove('spinner');
            // Update icon to checkmark
            buttonIcon.innerHTML = '<path d="M9,20.42L2.79,14.21L5.62,11.38L9,14.77L18.88,4.88L21.71,7.71L9,20.42Z"/>';
        }

        const signInSuccessEvent = new CustomEvent("NovioSignInSuccess", {
            bubbles: true,
            cancelable: false,
        });
        document.dispatchEvent(signInSuccessEvent);

        // Bind dropdown events for connected state
        this.bindDropdownEvents();
    }

    setLiveConnectionState(state) {

        const liveIndicator = this.shadowRoot.getElementById('liveIndicator');
        const liveIndicatorText = this.shadowRoot.getElementById('liveIndicatorText');

        liveIndicator.style.display = state ? 'inline-block' : 'none';
        liveIndicatorText.style.display = state ? 'inline-block' : 'none';

        liveIndicatorText.innerText = state;

        switch (state) {
            case 'live':
                liveIndicator.classList.remove('connecting');
                liveIndicator.classList.remove('disconnected');
                break;
            case 'connecting':
                liveIndicator.classList.add('connecting');
                break;
            case 'disconnected':
                liveIndicator.classList.add('disconnected');
                break;
            default:
                break;
        }
        if (liveIndicator === 'live') {
        }
    }

    setErrorState(error) {
        this.isConnecting = false;

        const button = this.shadowRoot.getElementById('connectButton');
        const buttonText = this.shadowRoot.getElementById('buttonText');
        const buttonIcon = this.shadowRoot.getElementById('buttonIcon');

        if (button) {
            button.disabled = false;
            button.classList.add('error');
        }

        if (buttonIcon) {
            buttonIcon.classList.remove('spinner');
        }

        if (error.message === 'WALLET_NOT_INSTALLED') {
            if (buttonText) buttonText.textContent = 'Install Novio';
            if (button) {
                button.onclick = () => window.open('https://chromewebstore.google.com/detail/novio/cabglodmdpffjfaoamokklhhidpilopi', '_blank');
            }
        } else {
            if (buttonText) buttonText.textContent = 'Connection Failed';
        }

        // Reset after 3 seconds
        setTimeout(() => {
            if (button) {
                button.classList.remove('error');
                button.onclick = null; // Clear the install link handler
            }
            if (buttonText) buttonText.textContent = 'Novio Login';
            if (buttonIcon) {
                buttonIcon.innerHTML = '<path d="M21 18v1c0 1.1-.9 2-2 2H5c-1.11 0-2-.9-2-2V5c0-1.1.89-2 2-2h14c1.1 0 2 .9 2 2v1h-9c-1.11 0-2 .9-2 2v8c0 1.1.89 2 2 2h9zm-9-2h10V8H12v8zm4-2.5c-.83 0-1.5-.67-1.5-1.5s.67-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5z"/>';
            }
            // Re-bind the original connect handler
            this.bindEvents();
        }, 3000);
    }

    // Complete reset method - this is what you want for sign out
    reset() {
        // Clean up all event listeners
        this.unbindEvents();

        // Reset all state
        this.isConnecting = false;
        this.isConnected = false;
        this.wallet = null;

        // Reset UI elements
        const button = this.shadowRoot.getElementById('connectButton');
        const buttonText = this.shadowRoot.getElementById('buttonText');
        const buttonIcon = this.shadowRoot.getElementById('buttonIcon');
        const menu = this.shadowRoot.getElementById('dropdownMenu');
        const liveIndicator = this.shadowRoot.getElementById('liveIndicator');
        const liveIndicatorText = this.shadowRoot.getElementById('liveIndicatorText');

        if (button) {
            button.disabled = false;
            button.classList.remove('connected', 'error', 'active');
            button.onclick = null;
        }

        if (buttonText) {
            buttonText.textContent = 'Novio Login';
        }

        if (buttonIcon) {
            buttonIcon.classList.remove('spinner');
            buttonIcon.innerHTML = '<path d="M21 18v1c0 1.1-.9 2-2 2H5c-1.11 0-2-.9-2-2V5c0-1.1.89-2 2-2h14c1.1 0 2 .9 2 2v1h-9c-1.11 0-2 .9-2 2v8c0 1.1.89 2 2 2h9zm-9-2h10V8H12v8zm4-2.5c-.83 0-1.5-.67-1.5-1.5s.67-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5z"/>';
        }

        if (menu) {
            menu.classList.remove('open');
        }

        if (liveIndicator) {
            liveIndicator.style.display = 'none';
            liveIndicatorText.style.display = 'none';
        }

        // Re-bind the initial connect events
        this.bindEvents();

        // Dispatch disconnected event
        this.dispatchEvent(new CustomEvent('novio-disconnected', {
            bubbles: true
        }));
    }

    // Public methods for developers
    async connect() {
        return this.handleConnect();
    }

    disconnect() {
        this.reset();
    }

    isWalletConnected() {
        return this.isConnected;
    }

    // Clean up when element is removed from DOM
    disconnectedCallback() {
        this.unbindEvents();
    }
}

// Register the custom element
customElements.define('novio-connect', NovioWalletButton);
