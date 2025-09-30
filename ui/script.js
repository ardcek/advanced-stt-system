// Advanced STT System Dashboard JavaScript
let currentTab = 'overview';

// Tab Management
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName).classList.add('active');
    
    // Add active class to clicked button
    event.target.classList.add('active');
    
    currentTab = tabName;
}

// File Upload Handler
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('audioFile');
    const uploadArea = document.getElementById('uploadArea');
    
    // Drag and drop functionality
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--primary-color)';
        uploadArea.style.background = 'rgba(37, 99, 235, 0.1)';
    });
    
    uploadArea.addEventListener('dragleave', function(e) {
        uploadArea.style.borderColor = 'var(--border-color)';
        uploadArea.style.background = 'var(--card-bg)';
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--border-color)';
        uploadArea.style.background = 'var(--card-bg)';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
    
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
});

// File Selection Handler
function handleFileSelect(file) {
    const uploadArea = document.getElementById('uploadArea');
    const allowedTypes = ['audio/wav', 'audio/mp3', 'audio/mp4', 'audio/m4a'];
    
    if (!allowedTypes.includes(file.type) && !file.name.match(/\.(wav|mp3|mp4|m4a)$/i)) {
        alert('Desteklenmeyen dosya formatı! WAV, MP3, MP4 veya M4A dosyası seçin.');
        return;
    }
    
    if (file.size > 100 * 1024 * 1024) { // 100MB limit
        alert('Dosya çok büyük! Maksimum 100MB boyutunda dosya yükleyebilirsiniz.');
        return;
    }
    
    // Update upload area
    uploadArea.innerHTML = `
        <i class="fas fa-file-audio"></i>
        <h3>Dosya Seçildi</h3>
        <p><strong>${file.name}</strong></p>
        <p>Boyut: ${formatFileSize(file.size)}</p>
        <button onclick="document.getElementById('audioFile').click()">Değiştir</button>
    `;
    
    // Enable process button
    document.querySelector('.process-btn').disabled = false;
    document.querySelector('.process-btn').style.opacity = '1';
}

// Format File Size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Process Audio (Demo)
function processAudio() {
    const fileInput = document.getElementById('audioFile');
    if (!fileInput.files.length) {
        alert('Lütfen önce bir ses dosyası seçin!');
        return;
    }
    
    const quality = document.getElementById('quality').value;
    const contentType = document.getElementById('contentType').value;
    const language = document.getElementById('language').value;
    
    // Show processing state
    const processBtn = document.querySelector('.process-btn');
    const originalText = processBtn.innerHTML;
    processBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> İşleniyor...';
    processBtn.disabled = true;
    
    // Simulate processing time based on quality
    const processingTimes = {
        'fastest': 2000,
        'balanced': 4000,
        'highest': 8000,
        'ultra': 15000
    };
    
    setTimeout(() => {
        showDemoResults(quality, contentType, language);
        processBtn.innerHTML = originalText;
        processBtn.disabled = false;
    }, processingTimes[quality] || 4000);
}

// Show Demo Results
function showDemoResults(quality, contentType, language) {
    const resultsDiv = document.getElementById('demoResults');
    const transcriptDiv = document.getElementById('transcriptResult');
    const accuracySpan = document.getElementById('accuracy');
    const timeSpan = document.getElementById('processingTime');
    
    // Sample transcripts based on content type
    const sampleTranscripts = {
        'general': 'Bu bir örnek transkripsiyon metnidir. Sistem ses dosyanızı yüksek doğrulukla yazıya çevirmiştir. Türkçe karakterler ve noktalama işaretleri doğru şekilde algılanmıştır.',
        'medical': 'Hasta şikayetleri: Başağrısı ve yorgunluk. Muayene: Kan basıncı 120/80 mmHg, nabız 72/dk, ateş 36.5°C. Tanı: Hipertansiyon. Tedavi: Lisinopril 10mg 1x1. Kontrol: 1 hafta sonra.',
        'academic': 'Bugünkü dersimizde makine öğrenmesi algoritmalarının temel prensiplerini inceleyeceğiz. Özellikle denetimli öğrenme yöntemleri ve bunların pratik uygulamalarına odaklanacağız. Regresyon analizi ve sınıflandırma problemlerini detaylı olarak ele alacağız.',
        'meeting': 'Toplantı başladı. Ahmet Bey: Proje durumu hakkında bilgi verebilir misiniz? Ayşe Hanım: Şu anda %80 tamamlandık. Mehmet Bey: Güzel, hedefe ulaşacağız gibi görünüyor. Karar: Haftalık raporlama devam edecek.'
    };
    
    // Accuracy based on quality
    const accuracyLevels = {
        'fastest': '87.3%',
        'balanced': '94.8%',
        'highest': '97.9%',
        'ultra': '99.91%'
    };
    
    // Processing time simulation
    const processingTimes = {
        'fastest': '0.8 saniye',
        'balanced': '2.3 saniye',
        'highest': '6.7 saniye',
        'ultra': '18.4 saniye'
    };
    
    transcriptDiv.innerHTML = `<p><strong>Transkripsiyon Sonucu:</strong></p><p style="background: var(--dark-bg); padding: 20px; border-radius: 10px; margin-top: 15px; line-height: 1.8;">${sampleTranscripts[contentType]}</p>`;
    
    accuracySpan.textContent = `Doğruluk: ${accuracyLevels[quality]}`;
    timeSpan.textContent = `İşleme Süresi: ${processingTimes[quality]}`;
    
    resultsDiv.style.display = 'block';
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
}

// Smooth Scrolling for Links
document.addEventListener('DOMContentLoaded', function() {
    // Add smooth scrolling to footer links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});

// Animate Stats on Scroll
function animateStats() {
    const statNumbers = document.querySelectorAll('.stat-content h3, .medical-stat .number, .accuracy, .score');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const target = entry.target;
                const text = target.textContent;
                const number = parseFloat(text.replace(/[^0-9.]/g, ''));
                
                if (!isNaN(number)) {
                    animateNumber(target, 0, number, 2000);
                }
                observer.unobserve(target);
            }
        });
    });
    
    statNumbers.forEach(stat => observer.observe(stat));
}

// Number Animation
function animateNumber(element, start, end, duration) {
    const range = end - start;
    const startTime = performance.now();
    const originalText = element.textContent;
    
    function updateNumber(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const current = start + (range * easeOutQuart(progress));
        
        // Preserve original formatting
        if (originalText.includes('%')) {
            element.textContent = current.toFixed(2) + '%';
        } else if (originalText.includes('+')) {
            element.textContent = Math.floor(current).toLocaleString() + '+';
        } else if (originalText.includes(',')) {
            element.textContent = Math.floor(current).toLocaleString();
        } else {
            element.textContent = current.toFixed(2);
        }
        
        if (progress < 1) {
            requestAnimationFrame(updateNumber);
        } else {
            element.textContent = originalText; // Restore original
        }
    }
    
    requestAnimationFrame(updateNumber);
}

// Easing function
function easeOutQuart(t) {
    return 1 - (--t) * t * t * t;
}

// Initialize animations when page loads
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(animateStats, 500); // Small delay for better effect
});

// Add loading states for better UX
function addLoadingStates() {
    // Add loading spinner for heavy operations
    const style = document.createElement('style');
    style.textContent = `
        .loading {
            position: relative;
            pointer-events: none;
        }
        .loading::after {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 20px;
            height: 20px;
            margin: -10px 0 0 -10px;
            border: 2px solid var(--primary-color);
            border-radius: 50%;
            border-top-color: transparent;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    `;
    document.head.appendChild(style);
}

// Initialize loading states
document.addEventListener('DOMContentLoaded', addLoadingStates);

// Add keyboard navigation
document.addEventListener('keydown', function(e) {
    if (e.key === 'Tab') {
        // Handle tab navigation
        document.body.classList.add('keyboard-nav');
    }
});

document.addEventListener('mousedown', function() {
    document.body.classList.remove('keyboard-nav');
});

// Add focus styles for keyboard navigation
const keyboardStyle = document.createElement('style');
keyboardStyle.textContent = `
    .keyboard-nav .tab-btn:focus,
    .keyboard-nav button:focus,
    .keyboard-nav select:focus {
        outline: 2px solid var(--primary-color);
        outline-offset: 2px;
    }
`;
document.head.appendChild(keyboardStyle);