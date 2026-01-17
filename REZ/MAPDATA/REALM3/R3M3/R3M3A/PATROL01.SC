Name "patrol01"

///////////////////////////////////////////////////////////////////////////////
//	set up some starting states for this script
///////////////////////////////////////////////////////////////////////////////

AbortOnDamage			FALSE		//	abort the script if we're damaged
AbortOnEnemySighted		FALSE		//	Don't abort if we sight an enemy
SightRange			1024.0		//	can see 1024 units
TimeOut				999.0
CheckForEnemyInterval		0.5		//	check for enemies every 1/2 second
TargetPlayer					//	target the player
YawRate				360.0
CanBeMoved			FALSE		//  don't allow outside forces to push the monster
Walk						//	walk to the path3 and wait

///////////////////////////////////////////////////////////////////////////////
//	Patrolling, not attacking...
///////////////////////////////////////////////////////////////////////////////

Loop1:
OnEnemyOutOfRange_Goto		DISABLE		//	don't branch on enemy out of range
OnTimerExpired_Goto		DISABLE	
OnEnemySighted_Goto		ATTACK1		//	if we see the player, jump to Attack
OnDamage_Goto			ATTACK1		//	if we get hit, jump to FarAttack
TimeOut				10.0
StartAnim				"Walk"
TurnTo				pat01pos01
MoveTo				pat01pos01
TurnToTarget			0.57		//	if the player is in range, attack
StartAnim	"AmbA"
Wait		1.0
Goto 	Loop2

///////////////////////////////////////////////////////////////////////////////

Loop2:
OnEnemyOutOfRange_Goto		DISABLE		//	don't branch on enemy out of range
OnTimerExpired_Goto		DISABLE	
OnEnemySighted_Goto		ATTACK2		//	if we see the player, jump to Attack
OnDamage_Goto			ATTACK2		//	if we get hit, jump to FarAttack
TimeOut				10.0
StartAnim				"Walk"
TurnTo				pat01pos02
MoveTo				pat01pos02
TurnToTarget			0.57		//	if the player is in range, attack
StartAnim	"AmbA"
Wait		1.0
Goto 	Loop3

///////////////////////////////////////////////////////////////////////////////

Loop3:
OnEnemyOutOfRange_Goto		DISABLE		//	don't branch on enemy out of range
OnTimerExpired_Goto		DISABLE	
OnEnemySighted_Goto		ATTACK3		//	if we see the player, jump to Attack
OnDamage_Goto			ATTACK3		//	if we get hit, jump to FarAttack
TimeOut				10.0
StartAnim				"Walk"
TurnTo				pat01pos03
MoveTo				pat01pos03
TurnToTarget			0.57		//	if the player is in range, attack
StartAnim	"AmbA"
Wait		1.0
Goto 	Loop4

///////////////////////////////////////////////////////////////////////////////

Loop4:
OnEnemyOutOfRange_Goto		DISABLE		//	don't branch on enemy out of range
OnTimerExpired_Goto		DISABLE	
OnEnemySighted_Goto		ATTACK4		//	if we see the player, jump to Attack
OnDamage_Goto			ATTACK4		//	if we get hit, jump to FarAttack
TimeOut				10.0
StartAnim				"Walk"
TurnTo				pat01pos04
MoveTo				pat01pos04
TurnToTarget			0.57		//	if the player is in range, attack
StartAnim	"AmbA"
Wait		1.0
Goto 	Loop1



///////////////////////////////////////////////////////////////////////////////
//	do the meanie, attacking stuff here
///////////////////////////////////////////////////////////////////////////////

Attack1:
OnDamage_Goto			DISABLE	//	don't branch if damaged while attacking
OnEnemySighted_Goto		DISABLE	//	don't branch if enemy sighted while attacking
OnEnemyOutOfRange_Goto		Loop1
If_HealthLessThanHalf_Goto	Crazy
SightRange	1024.0
YawRate			540
TimeOut			2.5

TurnToTarget		0.33	//	this is the time start of attack to firing in anim
RangedAttackTarget   	1.0
MeleeAttackTarget

Goto	Attack1

Attack2:
OnDamage_Goto			DISABLE	//	don't branch if damaged while attacking
OnEnemySighted_Goto		DISABLE	//	don't branch if enemy sighted while attacking
OnEnemyOutOfRange_Goto		Loop2
If_HealthLessThanHalf_Goto	Crazy
SightRange	1024.0
YawRate			540
TimeOut			2.5

TurnToTarget		0.33	//	this is the time start of attack to firing in anim
RangedAttackTarget   	1.0
MeleeAttackTarget

Goto	Attack2

Attack3:
OnDamage_Goto			DISABLE	//	don't branch if damaged while attacking
OnEnemySighted_Goto		DISABLE	//	don't branch if enemy sighted while attacking
OnEnemyOutOfRange_Goto		Loop3
If_HealthLessThanHalf_Goto	Crazy
SightRange	1024.0
YawRate			540
TimeOut			2.5

TurnToTarget		0.33	//	this is the time start of attack to firing in anim
RangedAttackTarget   	1.0
MeleeAttackTarget

Goto	Attack3

Attack4:
OnDamage_Goto			DISABLE	//	don't branch if damaged while attacking
OnEnemySighted_Goto		DISABLE	//	don't branch if enemy sighted while attacking
OnEnemyOutOfRange_Goto		Loop4
If_HealthLessThanHalf_Goto	Crazy
SightRange	1024.0
YawRate			540
TimeOut			2.5

TurnToTarget		0.33	//	this is the time start of attack to firing in anim
RangedAttackTarget   	1.0
MeleeAttackTarget

Goto	Attack4


///////////////////////////////////////////////////////////////////////////////
//	When ballbuster is almost dead he goes Crazy.
///////////////////////////////////////////////////////////////////////////////
Crazy:
	LoopAnim	FALSE
	OnDamage_Goto			DISABLE
	OnEnemySighted_Goto		DISABLE
	OnEnemyOutOfRange_Goto		Stand
	TurnToTarget			0.57	
	RangedAttackTarget
	MeleeAttackTarget
	Goto				Crazy

///////////////////////////////////////////////////////////////////////////////
//	When ballbuster is almost dead he goes Crazy.
///////////////////////////////////////////////////////////////////////////////
Stand:
	LoopAnim	FALSE
	OnDamage_Goto			crazy
	OnEnemySighted_Goto		crazy
	TurnToTarget			0.57	
	Goto				stand

