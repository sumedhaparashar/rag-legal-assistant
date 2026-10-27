
document.addEventListener('DOMContentLoaded', async () => {
    const loading = document.getElementById('admin-loading');
    const errorBox = document.getElementById('admin-error');
    const content = document.getElementById('admin-content');

    function showError(message) {
        loading.hidden = true;
        content.hidden = true;
        errorBox.textContent = message;
        errorBox.hidden = false;
    }

    try {
        // 1. Initialize Supabase using the backend configuration.
        const configResponse = await fetch('/api/auth/config');

        if (!configResponse.ok) {
            throw new Error('Unable to load authentication configuration.');
        }

        const config = await configResponse.json();

        if (!config.supabase_url || !config.supabase_anon_key) {
            throw new Error('Authentication is not configured.');
        }

        const client = window.supabase.createClient(
            config.supabase_url,
            config.supabase_anon_key
        );

        // 2. Retrieve the current Google login session.
        const { data: { session }, error: sessionError } =
            await client.auth.getSession();

        if (sessionError) throw sessionError;

        if (!session) {
            window.location.replace('/ui/auth.html');
            return;
        }

        document.getElementById('user-email').textContent =
            session.user.email || '';

        const headers = {
            Authorization: `Bearer ${session.access_token}`
        };

        // 3. Verify administrator access on the backend.
        const adminResponse = await fetch('/api/auth/me', {
            headers
        });

        if (!adminResponse.ok) {
            showError('Unable to verify administrator access.');
            return;
        }

        const adminData = await adminResponse.json();

        if (adminData.is_admin !== true) {
            showError('Access denied. This page is for administrators only.');
            return;
        }

        // 4. Retrieve protected analytics data.
        const analyticsResponse = await fetch('/api/analytics', {
            headers
        });

        if (!analyticsResponse.ok) {
            showError('Unable to load analytics data.');
            return;
        }

        const data = await analyticsResponse.json();

        // 5. Display real statistics.
        const setText = (id, value) => {
            document.getElementById(id).textContent = value;
        };

        setText('total-users', data.total_users);
        setText('daily-users', data.daily_active_users);
        setText('total-questions', data.total_questions);
        setText('weekly-questions', data.weekly_questions);

        setText(
            'response-time',
            (data.avg_response_time_ms / 1000).toFixed(2) + ' s'
        );

        const hours = Math.floor(data.uptime_seconds / 3600);
        const minutes = Math.floor((data.uptime_seconds % 3600) / 60);

        setText('uptime', `${hours}h ${minutes}m`);

        // 6. Render topic statistics safely.
        const topicsBody = document.getElementById('topics-body');

        for (const item of data.top_topics || []) {
            const row = topicsBody.insertRow();
            row.insertCell().textContent = item.topic;
            row.insertCell().textContent = item.count;
        }

        // 7. Render daily question counts.
        const dailyBody = document.getElementById('daily-body');

        for (const item of data.daily_question_counts || []) {
            const row = dailyBody.insertRow();
            row.insertCell().textContent = item.date;
            row.insertCell().textContent = item.count;
        }

        // 8. Show dashboard only after authorization succeeds.
        loading.hidden = true;
        content.hidden = false;

        // 9. Sign-out button.
        document.getElementById('btn-signout').addEventListener(
            'click',
            async () => {
                await client.auth.signOut();
                window.location.replace('/ui/auth.html');
            }
        );

    } catch (error) {
        console.error('Admin dashboard error:', error);
        showError('Unable to load the administrator dashboard.');
    }
});
