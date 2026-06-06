import NNetGA, { NNetGA as NNetGAClass } from '../nnetga.js';

class CartPole {
    constructor() {
        this.gravity = 9.8;
        this.masscart = 1.0;
        this.masspole = 0.1;
        this.total_mass = this.masspole + this.masscart;
        this.length = 2.0; // actually half the pole's length
        this.polemass_length = this.masspole * this.length;
        this.force_mag = 10.0;
        this.tau = 0.02; // seconds between state updates
        this.x_threshold = 50;
        this.theta_threshold_radians = 180 * Math.PI / 180;
        this.state = null;
        this.steps_beyond_done = null;
    }

    seed() { }

    reset() {
        // small random initial state
        this.state = [
            (Math.random() * 0.08) - 0.04,
            (Math.random() * 0.08) - 0.04,
            (Math.random() * 0.08) - 0.04,
            (Math.random() * 0.08) - 0.04
        ];
        this.steps_beyond_done = null;
        return this.state.slice();
    }

    step(action) {
        // action: 0 or 1
        let x = this.state[0];
        let x_dot = this.state[1];
        let theta = this.state[2];
        let theta_dot = this.state[3];

        const force = action === 1 ? this.force_mag : -this.force_mag;
        // const force = 0;
        const costheta = Math.cos(theta);
        const sintheta = Math.sin(theta);

        const temp = (force + this.polemass_length * theta_dot * theta_dot * sintheta) / this.total_mass;
        const thetaacc = (this.gravity * sintheta - costheta * temp) / (this.length * (4.0/3.0 - this.masspole * costheta * costheta / this.total_mass));
        const xacc = temp - this.polemass_length * thetaacc * costheta / this.total_mass;

        // semi-implicit (symplectic) Euler: update velocities first, then positions
        x_dot += this.tau * xacc;
        theta_dot += this.tau * thetaacc;

        // small damping to simulate friction and prevent energy growth
        const damping = 0.2; // per second damping coefficient
        x_dot *= (1 - damping * this.tau);
        theta_dot *= (1 - damping * this.tau);

        x += this.tau * x_dot;
        theta += this.tau * theta_dot;

        this.state = [x, x_dot, theta, theta_dot];

        const done = x < -this.x_threshold || x > this.x_threshold || theta < -this.theta_threshold_radians || theta > this.theta_threshold_radians;
        const reward = done ? 0 : 1;
        return { state: this.state.slice(), reward, done };
    }
}

// Utilities
function normalize(obs) {
    // obs: [x, x_dot, theta, theta_dot]
    const x = Math.max(Math.min(obs[0] / 2.4, 1), -1);
    const x_dot = Math.max(Math.min(obs[1] / 5, 1), -1);
    const theta = Math.max(Math.min(obs[2] / 0.21, 1), -1); // 12deg ~0.2094 rad
    const theta_dot = Math.max(Math.min(obs[3] / 5, 1), -1);
    return [x, x_dot, theta, theta_dot];
}

// Rendering
class Renderer {
    constructor(canvas, agentNum) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.agentNum = agentNum;
        this.width = canvas.width;
        this.height = canvas.height;
    }

    clear() {
        this.ctx.fillStyle = '#222';
        this.ctx.fillRect(0, 0, this.width, this.height);
    }

    drawAgents(states) {
        // states: array of [x, x_dot, theta, theta_dot]
        this.clear();
        const slotW = this.width / this.agentNum;
        const cartY = this.height * 0.75;
        const cartW = Math.min(40, slotW * 0.6);
        const cartH = 20;
        const poleLenPx = 120;

        for (let i = 0; i < this.agentNum; i++) {
            const s = states[i];
            const cx = (i + 0.5) * slotW;
            // map env x to small offset inside slot
            const xOffset = (s[0] / 2.4) * (slotW * 0.2);
            const cartX = cx + xOffset;

            // draw track
            this.ctx.strokeStyle = '#444';
            this.ctx.lineWidth = 1;
            this.ctx.beginPath();
            this.ctx.moveTo(cx - slotW*0.4, cartY + cartH/2 + 10);
            this.ctx.lineTo(cx + slotW*0.4, cartY + cartH/2 + 10);
            this.ctx.stroke();

            // cart
            this.ctx.fillStyle = '#aaa';
            this.ctx.fillRect(cartX - cartW/2, cartY - cartH/2, cartW, cartH);

            // pole
            const angle = s[2];
            const poleX = cartX;
            const poleY = cartY - cartH/2;
            const px2 = poleX + Math.sin(angle) * poleLenPx;
            const py2 = poleY - Math.cos(angle) * poleLenPx;
            this.ctx.strokeStyle = '#f5d02e';
            this.ctx.lineWidth = 4;
            this.ctx.beginPath();
            this.ctx.moveTo(poleX, poleY);
            this.ctx.lineTo(px2, py2);
            this.ctx.stroke();

            // small index
            this.ctx.fillStyle = '#fff';
            this.ctx.font = '10px sans-serif';
            this.ctx.fillText(i, cx - slotW*0.45 + 4, cartY - cartH - 6);
        }
    }
}

// Main integration
export function startCartPole(options = {}) {
    const agentNum = options.agentNum || 50;
    const generations = options.generations || 200;
    const maxSteps = options.maxSteps || 1000;

    const canvas = document.getElementById('cp-canvas');
    canvas.width = options.width || 1200;
    canvas.height = options.height || 600;

    const renderer = new Renderer(canvas, agentNum);

    // setup GA
    const nnetga = new NNetGAClass();
    nnetga.add_pop(1, 0.7, 0.02, 0.5);
    nnetga.add_agent(0, agentNum);
    nnetga.add_net(0, 4, 1, 5, 1);

    // environments per agent
    let envs = [...Array(agentNum)].map(() => new CartPole());
    let states = envs.map(e => e.reset());
    let scores = Array(agentNum).fill(0);

    let gen = 0;

    function runGeneration() {
        // reset
        envs = [...Array(agentNum)].map(() => new CartPole());
        states = envs.map(e => e.reset());
        scores = Array(agentNum).fill(0);
        let done = Array(agentNum).fill(false);
        let stepCount = 0;

        function stepFrame() {
            // build batch
            const batch = [states.map(s => normalize(s))];
            const outputs = nnetga.update(batch); // [pop][agent][out]
            const outs = outputs[0];

            for (let i = 0; i < agentNum; i++) {
                if (done[i]) continue;
                const out = outs[i][0];
                const action = out > 0.5 ? 1 : 0;
                const res = envs[i].step(action);
                states[i] = res.state;
                scores[i] += res.reward;
                if (res.done || stepCount >= maxSteps) {
                    done[i] = true;
                }
            }

            renderer.drawAgents(states);

            stepCount++;
            if (done.every(d => d) || stepCount >= maxSteps) {
                // finish generation
                gen += 1;
                // use scores as fitness
                nnetga.next_gen(0, scores, 1, 1, 1, 0);
                // log
                const avg = Math.round(scores.reduce((a,b)=>a+b,0)/agentNum);
                console.log(`Generation ${gen}  avg:${avg}  best:${Math.max(...scores)}`);
                if (gen < generations) {
                    requestAnimationFrame(runGeneration);
                } else {
                    console.log('Training finished');
                }
                return;
            }

            requestAnimationFrame(stepFrame);
        }

        requestAnimationFrame(stepFrame);
    }

    runGeneration();
}

// Auto-start if included directly
if (typeof document !== 'undefined' && document.readyState === 'complete') {
    startCartPole();
}
